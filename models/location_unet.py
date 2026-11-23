"""Learned location conditioning for the SD-1.5 inpainting U-Net: a heatmap input + LoRA.

    U-Net input  [z_t (4), mask (1), masked-image latent (4), location heatmap (1)]   (64 x 64)
    conv_in      9 -> 10 input channels; the new channel's weights start at zero, so the untrained model is
                 exactly the original one (how SD-inpainting itself was made from SD)
    multiscale   optionally the heatmap is also added after every down block (64 -> 8 resolution) through
                 zero-initialised 1x1 convolutions, as in T2I-Adapter; a single input channel learns too slowly
                 (its weights barely moved in 2k steps)
    attention    rank-r LoRA on to_q / to_k / to_v / to_out of every self- and cross-attention layer, so the
                 frozen U-Net can learn to use the heatmap

The heatmap is a Gaussian at where spatial audio places the source (models/spatial_pipeline.py), peak 1, all
zeros = "no location". The multiscale hooks read it from the 10th input channel, so callers only ever pass it
there.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler

LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) B A x with W frozen, B initialised to zero."""

    def __init__(self, base, rank=8, alpha=8.0):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * self.scale


def _add_multiscale(unet):
    """Zero-initialised 1x1 conv per down block that adds the (resized) heatmap to the block's output."""
    unet.location_injectors = nn.ModuleList(
        nn.Conv2d(1, block.resnets[-1].out_channels, 1) for block in unet.down_blocks)
    for conv in unet.location_injectors:
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)

    def grab(module, args):
        unet._location_heat = args[0][:, -1:]

    unet.conv_in.register_forward_pre_hook(grab)
    for block, conv in zip(unet.down_blocks, unet.location_injectors):
        def add(module, args, output, conv=conv):
            hidden, skips = output
            heat = F.interpolate(unet._location_heat.to(hidden.dtype), size=hidden.shape[-2:], mode="area")
            return hidden + conv(heat), skips

        block.register_forward_hook(add)


def add_location_inputs(unet, rank=8, alpha=8.0, multiscale=False):
    """Expand conv_in to 10 channels (zero-initialised), optionally add multiscale injectors, and wrap attention
    projections in LoRA. In place. Returns the list of trainable parameters; everything else is frozen."""
    unet.requires_grad_(False)
    old = unet.conv_in
    new = nn.Conv2d(old.in_channels + 1, old.out_channels, old.kernel_size, old.stride, old.padding)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old.in_channels] = old.weight
        new.bias.copy_(old.bias)
    unet.conv_in = new.to(old.weight.device, old.weight.dtype)
    unet.register_to_config(in_channels=old.in_channels + 1)
    if multiscale:
        _add_multiscale(unet)
        unet.location_injectors.to(old.weight.device, old.weight.dtype)
    attentions = [m for m in unet.modules() if type(m).__name__ == "Attention"]
    for module in attentions:
        for target in LORA_TARGETS:
            parent_name, _, child = target.rpartition(".")
            holder = module.get_submodule(parent_name) if parent_name else module
            if isinstance(getattr(holder, child, None), nn.Linear):
                setattr(holder, child, LoRALinear(getattr(holder, child), rank, alpha).to(old.weight.device))
    params = [p for n, p in unet.named_parameters() if is_location_param(n)]
    for p in params:
        p.requires_grad_(True)
    return params


def is_location_param(name):
    return name.startswith(("conv_in.", "location_injectors.")) or name.endswith((".down.weight", ".up.weight"))


def is_lora_param(name):
    return name.endswith((".down.weight", ".up.weight"))


def location_state_dict(unet):
    return {k: v.detach().cpu() for k, v in unet.state_dict().items() if is_location_param(k)}


def load_location_state(unet, state, rank=8, alpha=8.0):
    add_location_inputs(unet, rank, alpha, multiscale=any(k.startswith("location_injectors.") for k in state))
    missing, unexpected = unet.load_state_dict(state, strict=False)
    if unexpected or any(is_location_param(k) for k in missing):
        raise RuntimeError(f"location state does not match the U-Net: unexpected {unexpected[:3]}")
    unet.requires_grad_(False)
    return unet


def gaussian_heatmap(cx, cy, sx, sy, size, scale=1.0, device="cpu"):
    """Gaussian bump with peak 1 at (cx, cy), std (sx, sy), all given in pixels of a scale*size image,
    rendered on a size x size grid (size = 64 for the latent)."""
    ys = (torch.arange(size, device=device, dtype=torch.float32) + 0.5) * scale
    xs = (torch.arange(size, device=device, dtype=torch.float32) + 0.5) * scale
    gy = torch.exp(-0.5 * ((ys - cy) / max(sy, 1e-3)) ** 2)
    gx = torch.exp(-0.5 * ((xs - cx) / max(sx, 1e-3)) ** 2)
    return gy[:, None] * gx[None, :]


@torch.no_grad()
def sample_location(unet, vae, scheduler_config, cond, uncond, image, mask, heat, steps=50, guidance_scale=7.5,
                    seed=0):
    """DDIM inpainting with the 10-channel U-Net. image [B,3,R,R] in [-1,1], mask [B,1,R,R] (1 = fill),
    heat [B,1,R/8,R/8], cond/uncond [B,77,768]. The heatmap goes into both guidance branches (it is where,
    not what). Returns [B,3,R,R] in [-1,1] with the unmasked pixels pasted back."""
    device = image.device
    scale = vae.config.scaling_factor
    z_masked = vae.encode((image * (mask < 0.5)).to(vae.dtype)).latent_dist.mode().float() * scale
    m64 = F.interpolate(mask, size=z_masked.shape[-2:])
    sched = DDIMScheduler.from_config(scheduler_config)
    sched.set_timesteps(steps, device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    lat = torch.randn(z_masked.shape, generator=g, device=device) * sched.init_noise_sigma
    extra = torch.cat([m64, z_masked, heat], dim=1)
    for t in sched.timesteps:
        x = torch.cat([sched.scale_model_input(lat, t)] * 2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            eps = unet(torch.cat([x, torch.cat([extra] * 2)], dim=1), t,
                       encoder_hidden_states=torch.cat([uncond, cond])).sample.float()
        eps_u, eps_c = eps.chunk(2)
        lat = sched.step(eps_u + guidance_scale * (eps_c - eps_u), t, lat).prev_sample
    out = vae.decode((lat / scale).to(vae.dtype)).sample.float().clamp(-1, 1)
    return out * mask + image * (1 - mask)
