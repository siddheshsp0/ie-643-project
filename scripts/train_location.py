"""Train location conditioning (heatmap channel + LoRA) on AVSBench S4, on top of the frozen audio model.

Frozen: CLAP (features cached), the trained AudioAdapter (runs/s4_v1), CLIP, VAE, U-Net weights.
Trainable: the U-Net's heatmap input (conv_in channel, plus 1x1 injectors after every down block with
--multiscale) at --lr-location, and rank-8 LoRA on its attention layers at --lr. Timesteps are drawn with a
cubic bias toward high noise (T2I-Adapter), where the layout, and so the location, is decided.

Per step (dataloaders/avs_s4_location.py: a hole larger than the object + a heatmap where the object is):
    z0 = VAE(image), z_masked = VAE(image * (1 - hole)), m64 / h64 = hole / heatmap at latent size
    cond = CLIP("a photo of a" + adapter(CLAP feats))      (empty prompt with prob --cond-drop)
    eps_hat = UNet([z_t, m64, z_masked, h64], t, cond)
    loss = sum(w (eps_hat - eps)^2) / sum(w),  w = 1 + alpha * hole + beta * object

Validation every --val-every steps on --num-val val clips with small objects and the object at one end of a
3x wide hole, fixed noise and timesteps: masked MSE with the correct heatmap, with the heatmap mirrored to the
empty end of the hole, and with none. Success: correct < none < mirrored.

Usage (from the project root):
    # the kept model (runs/loc_v2b_multiscale); without --multiscale it is runs/loc_v2a_convin
    CUDA_VISIBLE_DEVICES=0 python scripts/train_location.py --run-name loc_v2b_multiscale --multiscale
    # smoke test
    CUDA_VISIBLE_DEVICES=0 python scripts/train_location.py --run-name loc_smoke --max-train-samples 16 \
        --max-steps 200 --warmup-steps 20 --num-val 16 --val-every 100 --sample-every 200 --save-every 200
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloaders.avs_s4 import label_text  # noqa: E402
from dataloaders.avs_s4_location import AVSS4LocationDataset  # noqa: E402
from models.audio_adapter import AudioAdapter  # noqa: E402
from models.audio_token_conditioner import DEFAULT_SD_PATH, AudioTokenConditioner  # noqa: E402
from models.location_unet import add_location_inputs, is_lora_param, location_state_dict, sample_location  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--out-root", default=str(ROOT / "runs"))
    p.add_argument("--adapter-ckpt", default=str(ROOT / "runs/s4_v1/last.pt"))
    p.add_argument("--sd-path", default=str(DEFAULT_SD_PATH))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--multiscale", action="store_true", help="also inject the heatmap after every down block")
    p.add_argument("--lr-location", type=float, default=1e-3, help="learning rate of the heatmap input weights")
    p.add_argument("--timesteps", choices=["cubic", "uniform"], default="cubic")
    p.add_argument("--val-max-width", type=float, default=0.33)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--cond-drop", type=float, default=0.1)
    p.add_argument("--hole-alpha", type=float, default=2.0, help="loss weight on the hole")
    p.add_argument("--object-beta", type=float, default=2.0, help="extra loss weight on the object")
    p.add_argument("--num-val", type=int, default=64)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--sample-every", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            batch[k] = [t.to(device, non_blocking=True) for t in v]
    return batch


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        self.out_dir = Path(args.out_root) / args.run_name
        (self.out_dir / "samples").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
        self.log_file = open(self.out_dir / "log.jsonl", "a")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.train_set = AVSS4LocationDataset("train", args.resolution, train=True,
                                              max_samples=args.max_train_samples)
        self.val_set = AVSS4LocationDataset("val", args.resolution, train=False, end_holes=True,
                                            max_object_width=args.val_max_width, max_samples=args.num_val)
        self.train_loader = DataLoader(self.train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                       num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

        dev = self.device
        self.vae = AutoencoderKL.from_pretrained(args.sd_path, subfolder="vae", variant="fp16",
                                                 torch_dtype=torch.float16).to(dev).eval().requires_grad_(False)
        self.unet = UNet2DConditionModel.from_pretrained(args.sd_path, subfolder="unet", variant="fp16",
                                                         torch_dtype=torch.float32).to(dev)
        self.params = add_location_inputs(self.unet, rank=args.lora_rank, alpha=float(args.lora_rank),
                                          multiscale=args.multiscale)
        named = [(n, p) for n, p in self.unet.named_parameters() if p.requires_grad]
        self.param_groups = [{"params": [p for n, p in named if not is_lora_param(n)], "lr": args.lr_location},
                             {"params": [p for n, p in named if is_lora_param(n)], "lr": args.lr}]
        self.unet.enable_gradient_checkpointing()
        self.unet.train()
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")
        ckpt = torch.load(args.adapter_ckpt, map_location=dev)
        self.adapter = AudioAdapter(**ckpt["adapter_config"]).to(dev).eval().requires_grad_(False)
        self.adapter.load_state_dict(ckpt["adapter"])
        self.conditioner = AudioTokenConditioner(args.sd_path, template=ckpt["template"],
                                                 num_tokens=ckpt["adapter_config"]["num_tokens"], device=dev)
        with torch.no_grad():
            tok = self.conditioner.tokenizer
            empty = tok("", padding="max_length", max_length=tok.model_max_length, return_tensors="pt").input_ids
            self.uncond = self.conditioner.text_encoder(input_ids=empty.to(dev)).last_hidden_state
        self.scale = self.vae.config.scaling_factor

        self.optimizer = torch.optim.AdamW(self.param_groups, weight_decay=args.weight_decay)
        self.lr_scheduler = get_scheduler("cosine", self.optimizer, num_warmup_steps=args.warmup_steps,
                                          num_training_steps=args.max_steps)
        self.step = 0
        n = sum(p.numel() for p in self.params)
        print(f"train clips {len(self.train_set)}  val clips {len(self.val_set)}  trainable params {n:,}", flush=True)
        g = torch.Generator().manual_seed(1234)
        lat = args.resolution // 8
        self.val_noise = torch.randn(len(self.val_set), 4, lat, lat, generator=g)
        self.val_t = torch.linspace(50, 950, len(self.val_set)).round().long()[
            torch.randperm(len(self.val_set), generator=g)]

    @torch.no_grad()
    def latents(self, batch, sample):
        def enc(x):
            dist = self.vae.encode(x.half()).latent_dist
            return (dist.sample() if sample else dist.mode()).float() * self.scale

        z0 = enc(batch["image"])
        z_masked = enc(batch["image"] * (batch["mask"] < 0.5))
        m64 = F.interpolate(batch["mask"], size=z0.shape[-2:])
        o64 = F.interpolate(batch["obj"], size=z0.shape[-2:])
        return z0, z_masked, m64, o64

    @torch.no_grad()
    def audio_context(self, batch):
        return self.conditioner(self.adapter(batch["intermediates"], batch["global_embed"]))

    def eps(self, z_t, m64, z_masked, heat, t, cond):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.unet(torch.cat([z_t, m64, z_masked, heat], 1), t, encoder_hidden_states=cond).sample.float()

    def weighted_mse(self, pred, target, m64, o64):
        w = (1.0 + self.args.hole_alpha * m64 + self.args.object_beta * o64).expand_as(pred)
        return ((pred - target) ** 2 * w).sum() / w.sum()

    def train_step(self, batch):
        z0, z_masked, m64, o64 = self.latents(batch, sample=True)
        B = z0.shape[0]
        noise = torch.randn_like(z0)
        T = self.noise_scheduler.config.num_train_timesteps
        if self.args.timesteps == "cubic":  # mostly high noise, where the layout is decided
            t = ((1 - torch.rand(B, device=self.device) ** 3) * T).long().clamp(max=T - 1)
        else:
            t = torch.randint(0, T, (B,), device=self.device)
        z_t = self.noise_scheduler.add_noise(z0, noise, t)
        cond = self.audio_context(batch)
        drop = torch.rand(B, device=self.device) < self.args.cond_drop
        cond = torch.where(drop[:, None, None], self.uncond.expand(B, -1, -1), cond)
        loss = self.weighted_mse(self.eps(z_t, m64, z_masked, batch["heat"], t, cond), noise, m64, o64)
        return loss

    def train(self):
        a = self.args
        it = iter(self.train_loader)
        if self.step == 0:
            self.validate()
        running, t_last = [], time.time()
        while self.step < a.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(a.grad_accum):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(self.train_loader)
                    batch = next(it)
                loss = self.train_step(to_device(batch, self.device))
                (loss / a.grad_accum).backward()
                total += loss.item() / a.grad_accum
            gnorm = torch.nn.utils.clip_grad_norm_(self.params, a.max_grad_norm).item()
            if not math.isfinite(gnorm):
                raise RuntimeError(f"non-finite gradient at step {self.step}")
            self.optimizer.step()
            self.lr_scheduler.step()
            self.step += 1
            running.append((total, gnorm))
            if self.step % a.log_every == 0:
                dt = time.time() - t_last
                rec = {"step": self.step, "loss": float(np.mean([r[0] for r in running])),
                       "grad_norm": float(np.mean([r[1] for r in running])), "lr": self.lr_scheduler.get_last_lr()[0],
                       "steps_per_s": len(running) / dt, "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
                self.log(rec)
                print(f"step {self.step:>6}  loss {rec['loss']:.4f}  gnorm {rec['grad_norm']:.3f}  lr {rec['lr']:.2e}  "
                      f"{rec['steps_per_s']:.2f} it/s  {rec['mem_gb']:.1f} GB", flush=True)
                running, t_last = [], time.time()
            if self.step % a.val_every == 0 or self.step == a.max_steps:
                self.validate()
            if self.step % a.sample_every == 0 or self.step == a.max_steps:
                self.sample_grid()
            if self.step % a.save_every == 0 or self.step == a.max_steps:
                self.save()
            if self.step % a.log_every == 0:
                t_last = time.time()

    @torch.no_grad()
    def validate(self):
        self.unet.eval()
        loader = DataLoader(self.val_set, batch_size=self.args.batch_size, shuffle=False, num_workers=2)
        sums = {"correct": 0.0, "mirrored": 0.0, "none": 0.0}
        n = idx = 0
        for batch in loader:
            batch = to_device(batch, self.device)
            z0, z_masked, m64, o64 = self.latents(batch, sample=False)
            B = z0.shape[0]
            noise = self.val_noise[idx:idx + B].to(self.device)
            t = self.val_t[idx:idx + B].to(self.device)
            idx += B
            z_t = self.noise_scheduler.add_noise(z0, noise, t)
            cond = self.audio_context(batch)
            for k, heat in (("correct", batch["heat"]), ("mirrored", batch["heat_shift"]),
                            ("none", torch.zeros_like(batch["heat"]))):
                sums[k] += self.weighted_mse(self.eps(z_t, m64, z_masked, heat, t, cond), noise, m64, o64).item() * B
            n += B
        rec = {"step": self.step, **{f"val_mse_{k}": v / n for k, v in sums.items()}}
        self.log(rec)
        print(f"[val step {self.step}] mse correct {rec['val_mse_correct']:.5f}  none {rec['val_mse_none']:.5f}  "
              f"mirrored {rec['val_mse_mirrored']:.5f}", flush=True)
        self.unet.train()

    @torch.no_grad()
    def sample_grid(self, k=4):
        """Rows: val clips. Columns: masked input | heatmap | correct location | mirrored location | ground truth."""
        self.unet.eval()
        items = [self.val_set[i] for i in range(min(k, len(self.val_set)))]
        batch = to_device(torch.utils.data.default_collate(items), self.device)
        cond = self.audio_context(batch)
        uncond = self.uncond.expand(len(items), -1, -1)
        cfg = self.noise_scheduler.config
        good = sample_location(self.unet, self.vae, cfg, cond, uncond, batch["image"], batch["mask"], batch["heat"],
                               steps=30)
        bad = sample_location(self.unet, self.vae, cfg, cond, uncond, batch["image"], batch["mask"],
                              batch["heat_shift"], steps=30)
        tile = self.args.resolution // 2
        canvas = Image.new("RGB", (5 * tile, len(items) * (tile + 20)), "white")
        d = ImageDraw.Draw(canvas)

        def pil(x):
            return Image.fromarray(((x.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8))

        for r, it in enumerate(items):
            y = r * (tile + 20)
            d.text((4, y + 4), f"{label_text(self.val_set.classes[it['class_idx']])}   [masked | heatmap | correct "
                               f"location | mirrored location | ground truth]", fill="black")
            masked = batch["image"][r] * (batch["mask"][r] < 0.5)
            heat = F.interpolate(batch["heat"][r:r + 1], size=batch["image"].shape[-2:], mode="bilinear")[0]
            overlay = masked * (1 - heat) + torch.tensor([1.0, -0.6, -0.6], device=self.device)[:, None, None] * heat
            for c, x in enumerate([masked, overlay.clamp(-1, 1), good[r], bad[r], batch["image"][r]]):
                canvas.paste(pil(x).resize((tile, tile), Image.BICUBIC), (c * tile, y + 20))
        path = self.out_dir / "samples" / f"step_{self.step:06d}.png"
        canvas.save(path)
        print(f"[samples] {path}", flush=True)
        self.unet.train()

    def log(self, rec):
        self.log_file.write(json.dumps(rec) + "\n")
        self.log_file.flush()

    def save(self):
        ckpt = {"step": self.step, "location": location_state_dict(self.unet), "lora_rank": self.args.lora_rank,
                "multiscale": self.args.multiscale,
                "adapter_ckpt": self.args.adapter_ckpt, "args": vars(self.args)}
        torch.save(ckpt, self.out_dir / f"step_{self.step:06d}.pt")
        torch.save(ckpt, self.out_dir / "last.pt")
        print(f"[ckpt] step {self.step} -> {self.out_dir}", flush=True)


if __name__ == "__main__":
    Trainer(parse_args()).train()
