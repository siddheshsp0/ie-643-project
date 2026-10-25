"""Train the AudioAdapter so SD-1.5 inpainting can be conditioned on sound (AVSBench S4).

Only the adapter is trained; CLAP (features are cached), the VAE, CLIP and the U-Net stay frozen.
The loss is the usual noise-prediction MSE, weighted towards the masked region, plus a small L1
penalty on the audio tokens as in AudioToken (the classification term is off by default). The audio
prompt is swapped for the empty one with probability --cond-drop so CFG works at inference.

Usage (from the project root):
    CUDA_VISIBLE_DEVICES=1 python scripts/train_audio_adapter.py --run-name s4_v1
    # smoke test: overfit 16 clips
    CUDA_VISIBLE_DEVICES=1 python scripts/train_audio_adapter.py --run-name smoke --max-train-samples 16 \
        --max-steps 300 --warmup-steps 20 --num-val 16 --val-every 100 --sample-every 300 --save-every 300
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
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloaders.avs_s4 import AVSS4Dataset, label_text  # noqa: E402
from models.audio_adapter import AudioAdapter  # noqa: E402
from models.audio_token_conditioner import DEFAULT_SD_PATH, AudioTokenConditioner  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--out-root", default=str(ROOT / "runs"))
    p.add_argument("--sd-path", default=str(DEFAULT_SD_PATH))
    p.add_argument("--resume", default=None, help="checkpoint .pt to resume from")
    p.add_argument("--seed", type=int, default=0)
    # data
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--mask-probs", type=float, nargs=3, default=(0.6, 0.2, 0.2), metavar=("OBJ", "BOX", "FULL"))
    p.add_argument("--dilate-px", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    # adapter
    p.add_argument("--num-tokens", type=int, default=8)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--no-source-embeddings", action="store_true")
    p.add_argument("--template", default="a photo of a")
    # optimisation
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=20000, help="optimizer steps")
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--lr-schedule", default="cosine")
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--cond-drop", type=float, default=0.1, help="prob. of empty-prompt conditioning (for CFG)")
    p.add_argument("--mask-loss-alpha", type=float, default=2.0, help="loss weight 1 + alpha * mask")
    p.add_argument("--l1-weight", type=float, default=1e-3)
    p.add_argument("--cl-weight", type=float, default=0.0, help="AudioToken classification loss weight")
    # evaluation / logging
    p.add_argument("--num-val", type=int, default=64)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--sample-every", type=int, default=2000)
    p.add_argument("--num-sample", type=int, default=4, help="clips per split in the sample grid")
    p.add_argument("--sample-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device(batch, device):
    batch["intermediates"] = [t.to(device, non_blocking=True) for t in batch["intermediates"]]
    for k in ("global_embed", "image", "mask", "class_idx"):
        batch[k] = batch[k].to(device, non_blocking=True)
    return batch


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("training requires CUDA")
        self.out_dir = Path(args.out_root) / args.run_name
        (self.out_dir / "samples").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
        self.log_file = open(self.out_dir / "log.jsonl", "a")
        seed_everything(args.seed)

        # ---- data
        self.train_set = AVSS4Dataset("train", args.resolution, train=True, mask_probs=args.mask_probs,
                                      dilate_px=args.dilate_px, max_samples=args.max_train_samples)
        self.val_set = AVSS4Dataset("val", args.resolution, train=False, dilate_px=args.dilate_px,
                                    max_samples=args.num_val)
        self.train_eval_set = AVSS4Dataset("train", args.resolution, train=False, dilate_px=args.dilate_px,
                                           max_samples=args.max_train_samples)
        self.train_loader = DataLoader(self.train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                       num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
        self.classes = self.train_set.classes

        # ---- frozen models
        dev = self.device
        self.vae = AutoencoderKL.from_pretrained(args.sd_path, subfolder="vae", variant="fp16",
                                                 torch_dtype=torch.float16).to(dev).eval().requires_grad_(False)
        self.unet = UNet2DConditionModel.from_pretrained(args.sd_path, subfolder="unet", variant="fp16",
                                                         torch_dtype=torch.float32).to(dev)
        self.unet.requires_grad_(False)
        self.unet.enable_gradient_checkpointing()
        self.unet.train()  # checkpointing is active only in train mode; SD's U-Net has no dropout
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")
        self.conditioner = AudioTokenConditioner(args.sd_path, template=args.template,
                                                 num_tokens=args.num_tokens, device=dev)
        self.vae_scale = self.vae.config.scaling_factor

        with torch.no_grad():
            tok = self.conditioner.tokenizer
            empty = tok("", padding="max_length", max_length=tok.model_max_length, return_tensors="pt").input_ids
            self.uncond = self.conditioner.text_encoder(input_ids=empty.to(dev)).last_hidden_state  # [1, 77, 768]
            emb = self.conditioner.token_embedding
            # Mean CLIP word embedding of each class label (AudioToken classification-loss target).
            self.label_emb = torch.stack([
                emb(torch.tensor(tok(label_text(c), add_special_tokens=False).input_ids, device=dev)).mean(0)
                for c in self.classes])
            self.word_emb_norm = emb.weight.norm(dim=-1).mean().item()

        # ---- trainable adapter
        feat_dims = [t.shape[-1] for t in self.train_set.intermediates]
        global_dim = self.train_set.global_embed.shape[-1]
        self.adapter = AudioAdapter(feat_dims, global_dim, hidden_dim=args.hidden_dim, num_tokens=args.num_tokens,
                                    num_heads=args.num_heads,
                                    use_source_embeddings=not args.no_source_embeddings).to(dev)
        self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.lr_scheduler = get_scheduler(args.lr_schedule, self.optimizer, num_warmup_steps=args.warmup_steps,
                                          num_training_steps=args.max_steps)
        self.step = 0
        if args.resume:
            self.load(args.resume)

        n_params = sum(p.numel() for p in self.adapter.parameters())
        print(f"train clips {len(self.train_set)}  val clips {len(self.val_set)}  adapter params {n_params:,}")
        print(f"layers {self.train_set.layer_names} dims {feat_dims}  -> run dir {self.out_dir}")

        # Fixed noise/timesteps so validation losses are comparable across checkpoints.
        g = torch.Generator().manual_seed(1234)
        lat = args.resolution // 8
        self.val_noise = torch.randn(len(self.val_set), 4, lat, lat, generator=g)
        self.val_t = torch.linspace(50, 950, len(self.val_set)).round().long()[torch.randperm(len(self.val_set), generator=g)]
        self.val_empty_mse = None

    # ------------------------------------------------------------------ helpers
    @torch.no_grad()
    def encode(self, x, sample):
        dist = self.vae.encode(x.half()).latent_dist
        z = dist.sample() if sample else dist.mode()
        return z.float() * self.vae_scale

    @torch.no_grad()
    def prepare_latents(self, batch, sample=True):
        image, mask = batch["image"], batch["mask"]
        z0 = self.encode(image, sample)
        z_masked = self.encode(image * (mask < 0.5), sample)
        m64 = F.interpolate(mask, size=z0.shape[-2:])  # nearest, as in diffusers' inpaint pipeline
        return z0, z_masked, m64

    def unet_eps(self, z_t, m64, z_masked, t, cond):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.unet(torch.cat([z_t, m64, z_masked], dim=1), t, encoder_hidden_states=cond).sample
        return out.float()

    def weighted_mse(self, pred, target, m64):
        w = (1.0 + self.args.mask_loss_alpha * m64).expand_as(pred)
        return ((pred - target) ** 2 * w).sum() / w.sum()

    # ------------------------------------------------------------------ training
    def train_step(self, batch):
        a = self.args
        z0, z_masked, m64 = self.prepare_latents(batch, sample=True)
        B = z0.shape[0]
        noise = torch.randn_like(z0)
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device)
        z_t = self.noise_scheduler.add_noise(z0, noise, t)

        tokens = self.adapter(batch["intermediates"], batch["global_embed"])
        cond = self.conditioner(tokens)
        drop = torch.rand(B, device=self.device) < a.cond_drop
        cond = torch.where(drop[:, None, None], self.uncond.expand(B, -1, -1), cond)

        mse = self.weighted_mse(self.unet_eps(z_t, m64, z_masked, t, cond), noise, m64)
        l1 = tokens.abs().mean()
        cl = (1 - F.cosine_similarity(tokens.mean(1), self.label_emb[batch["class_idx"]], dim=-1)).mean()
        loss = mse + a.l1_weight * l1 + a.cl_weight * cl
        stats = {"loss": loss.item(), "mse": mse.item(), "l1": l1.item(), "cl": cl.item(),
                 "token_norm": tokens.norm(dim=-1).mean().item()}
        return loss, stats

    def train(self):
        a = self.args
        loader_iter = iter(self.train_loader)
        self.adapter.train()
        running, t_last = [], time.time()
        if self.step == 0:
            self.validate()
        while self.step < a.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            micro = []
            for _ in range(a.grad_accum):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(self.train_loader)
                    batch = next(loader_iter)
                loss, stats = self.train_step(to_device(batch, self.device))
                (loss / a.grad_accum).backward()
                micro.append(stats)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), a.max_grad_norm).item()
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient norm at step {self.step}")
            self.optimizer.step()
            self.lr_scheduler.step()
            self.step += 1
            running.append({k: sum(m[k] for m in micro) / len(micro) for k in micro[0]} | {"grad_norm": grad_norm})

            if self.step % a.log_every == 0 or self.step == a.max_steps:
                dt = time.time() - t_last
                avg = {k: sum(r[k] for r in running) / len(running) for k in running[0]}
                rec = {"step": self.step, "lr": self.lr_scheduler.get_last_lr()[0],
                       "steps_per_s": len(running) / dt, "mem_gb": torch.cuda.max_memory_allocated() / 2**30, **avg}
                self.log(rec)
                print(f"step {self.step:>6}  loss {avg['loss']:.4f}  mse {avg['mse']:.4f}  l1 {avg['l1']:.3f}  "
                      f"cl {avg['cl']:.3f}  |tok| {avg['token_norm']:.2f} (CLIP words {self.word_emb_norm:.2f})  "
                      f"gnorm {avg['grad_norm']:.3f}  lr {rec['lr']:.2e}  {rec['steps_per_s']:.2f} it/s  "
                      f"{rec['mem_gb']:.1f} GB")
                running, t_last = [], time.time()
            if self.step % a.val_every == 0 or self.step == a.max_steps:
                self.validate()
            if self.step % a.sample_every == 0 or self.step == a.max_steps:
                self.sample_grid()
            if self.step % a.save_every == 0 or self.step == a.max_steps:
                self.save()
            if self.step % a.log_every == 0:
                t_last = time.time()  # exclude validation/sampling time from throughput

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def validate(self):
        """Mask-weighted val MSE with fixed noise/timesteps for: audio, shuffled audio, empty prompt."""
        self.adapter.eval()
        loader = DataLoader(self.val_set, batch_size=self.args.batch_size, shuffle=False, num_workers=2)
        sums = {"audio": 0.0, "shuffled": 0.0, "empty": 0.0}
        n, idx = 0, 0
        for batch in loader:
            batch = to_device(batch, self.device)
            z0, z_masked, m64 = self.prepare_latents(batch, sample=False)
            B = z0.shape[0]
            noise = self.val_noise[idx:idx + B].to(self.device)
            t = self.val_t[idx:idx + B].to(self.device)
            idx += B
            z_t = self.noise_scheduler.add_noise(z0, noise, t)
            conds = {"audio": self.conditioner(self.adapter(batch["intermediates"], batch["global_embed"]))}
            if B > 1:  # pair each frame with another clip's audio
                perm = torch.roll(torch.arange(B, device=self.device), 1)
                conds["shuffled"] = conds["audio"][perm]
            if self.val_empty_mse is None:  # adapter-independent, compute once
                conds["empty"] = self.uncond.expand(B, -1, -1)
            for k, c in conds.items():
                sums[k] += self.weighted_mse(self.unet_eps(z_t, m64, z_masked, t, c), noise, m64).item() * B
            n += B
        if self.val_empty_mse is None:
            self.val_empty_mse = sums["empty"] / n
        rec = {"step": self.step, "val_mse_audio": sums["audio"] / n, "val_mse_shuffled": sums["shuffled"] / n,
               "val_mse_empty": self.val_empty_mse}
        self.log(rec)
        print(f"[val step {self.step}] mse audio {rec['val_mse_audio']:.4f}  shuffled {rec['val_mse_shuffled']:.4f}  "
              f"empty-prompt {rec['val_mse_empty']:.4f}")
        self.adapter.train()

    @torch.no_grad()
    def generate(self, batch, seed=0, use_audio=True):
        a = self.args
        self.adapter.eval()
        _, z_masked, m64 = self.prepare_latents(batch, sample=False)
        B = z_masked.shape[0]
        uncond = self.uncond.expand(B, -1, -1)
        cond = self.conditioner(self.adapter(batch["intermediates"], batch["global_embed"])) if use_audio else uncond
        cond_in = torch.cat([uncond, cond])
        sched = DDIMScheduler.from_pretrained(a.sd_path, subfolder="scheduler")
        sched.set_timesteps(a.sample_steps, device=self.device)
        g = torch.Generator(device=self.device).manual_seed(seed)
        lat = torch.randn(z_masked.shape, generator=g, device=self.device) * sched.init_noise_sigma
        m_in, zm_in = torch.cat([m64] * 2), torch.cat([z_masked] * 2)
        for t in sched.timesteps:
            x = sched.scale_model_input(torch.cat([lat] * 2), t)
            eps_u, eps_c = self.unet_eps(x, m_in, zm_in, t, cond_in).chunk(2)
            lat = sched.step(eps_u + a.guidance_scale * (eps_c - eps_u), t, lat).prev_sample
        img = self.vae.decode((lat / self.vae_scale).half()).sample.float().clamp(-1, 1)
        self.adapter.train()
        return img

    @torch.no_grad()
    def sample_grid(self):
        """One row per clip (train, then val): masked input | audio sample | ground truth | no-audio sample."""
        k = self.args.num_sample
        rows = []
        for name, ds in (("train", self.train_eval_set), ("val", self.val_set)):
            items = [ds[i] for i in range(min(k, len(ds)))]
            batch = to_device(torch.utils.data.default_collate(items), self.device)
            gen = self.generate(batch)
            gen_empty = self.generate(batch, use_audio=False)
            masked = batch["image"] * (batch["mask"] < 0.5)
            for i, it in enumerate(items):
                rows.append((f"{name}: {label_text(self.classes[it['class_idx']])}   "
                             f"[masked | audio | ground truth | no audio]",
                             [masked[i], gen[i], batch["image"][i], gen_empty[i]]))
        tile = self.args.resolution // 2
        canvas = Image.new("RGB", (4 * tile, len(rows) * (tile + 20)), "white")
        draw = ImageDraw.Draw(canvas)
        for r, (caption, imgs) in enumerate(rows):
            y = r * (tile + 20)
            draw.text((4, y + 4), caption, fill="black")
            for c, x in enumerate(imgs):
                arr = ((x.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                canvas.paste(Image.fromarray(arr).resize((tile, tile), Image.BICUBIC), (c * tile, y + 20))
        path = self.out_dir / "samples" / f"step_{self.step:06d}.png"
        canvas.save(path)
        print(f"[samples] {path}")

    # ------------------------------------------------------------------ io
    def log(self, rec):
        self.log_file.write(json.dumps(rec) + "\n")
        self.log_file.flush()

    def save(self):
        ckpt = {"step": self.step, "adapter_config": self.adapter.config, "adapter": self.adapter.state_dict(),
                "optimizer": self.optimizer.state_dict(), "lr_scheduler": self.lr_scheduler.state_dict(),
                "layer_names": self.train_set.layer_names, "template": self.args.template, "args": vars(self.args)}
        torch.save(ckpt, self.out_dir / f"step_{self.step:06d}.pt")
        torch.save(ckpt, self.out_dir / "last.pt")
        print(f"[ckpt] saved step {self.step} -> {self.out_dir}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        if ckpt["adapter_config"] != self.adapter.config:
            raise ValueError(f"adapter config mismatch: {ckpt['adapter_config']} vs {self.adapter.config}")
        self.adapter.load_state_dict(ckpt["adapter"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        self.step = ckpt["step"]
        print(f"resumed from {path} at step {self.step}")


if __name__ == "__main__":
    Trainer(parse_args()).train()
