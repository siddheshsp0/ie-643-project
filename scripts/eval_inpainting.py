"""Test-set evaluation of audio-conditioned inpainting (Approach 1) on AVSBench S4.

For every S4 test clip, frame 1 is inpainted inside its dilated ground-truth sounding-object mask
under four conditionings, with the same seed and mask, so the images differ only in the conditioning:

    audio     the clip's own audio (CLAP -> AudioAdapter -> CLIP "a photo of a * x 8")
    shuffled  the audio of a test clip from a different class (wrong audio)
    none      empty prompt: plain SD-1.5 inpainting, the visual-only baseline
    text      "a photo of a <class label>": oracle text prompt, an upper bound for semantic guidance

    generate  run the pipeline (shardable over GPUs, skips images that already exist)
    report    score every image against the original frame, then write metrics.csv, summary.json,
              README.md and a qualitative grid

Scores (generated pixels are pasted into the original frame with the mask):
    PSNR      over the masked pixels only
    SSIM      on the mask's bounding-box crop
    LPIPS     (AlexNet) on the mask's bounding-box crop resized to 256 x 256
    FD-CLIP   Frechet distance between CLIP ViT-L/14 image features of generated and real region crops
    CLIP acc  CLIP ViT-L/14 zero-shot top-1 over the 23 S4 classes on the generated region (bounding box,
              everything outside the mask gray), against the image's class; for 'shuffled' also against
              the class of the audio that was used

Usage (from the project root):
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_inpainting.py generate --shard 0 --num-shards 2 &
    CUDA_VISIBLE_DEVICES=1 python scripts/eval_inpainting.py generate --shard 1 --num-shards 2
    python scripts/eval_inpainting.py report
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloaders.avs_s4 import (CACHE_DIR, all_classes, class_balanced_subset, frame_path,  # noqa: E402
                                label_text, mask_path, read_split)
from models.audio_token_conditioner import DEFAULT_SD_PATH  # noqa: E402

CONDITIONS = ["audio", "shuffled", "none", "text"]
CLIP_PATH = ROOT / "models/clip-vit-large-patch14"
OBJECT_NOUNS = {  # CLIP prompt "a photo of {noun}." per S4 class (same as eval_novel_view.py)
    "ambulance_siren": "an ambulance", "baby_laughter": "a baby", "cap_gun_shooting": "a toy gun",
    "cat_meowing": "a cat", "chainsawing_trees": "a chainsaw", "coyote_howling": "a coyote",
    "dog_barking": "a dog", "driving_buses": "a bus", "female_singing": "a woman singing",
    "helicopter": "a helicopter", "horse_clip-clop": "a horse", "lawn_mowing": "a lawn mower",
    "lions_roaring": "a lion", "male_speech": "a man speaking", "mynah_bird_singing": "a bird",
    "playing_acoustic_guitar": "an acoustic guitar", "playing_glockenspiel": "a glockenspiel",
    "playing_piano": "a piano", "playing_tabla": "tabla drums", "playing_ukulele": "a ukulele",
    "playing_violin": "a violin", "race_car": "a race car", "typing_on_computer_keyboard": "a computer keyboard",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["generate", "report"])
    p.add_argument("--out", default=str(ROOT / "outputs/inpainting_eval"))
    p.add_argument("--ckpt", default=str(ROOT / "runs/s4_v1/last.pt"), help="audio adapter checkpoint")
    p.add_argument("--max-clips", type=int, default=None, help="class-balanced subset of the test split")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--dilate-px", type=int, default=8, help="mask dilation (training used 8)")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--sd-path", default=str(DEFAULT_SD_PATH))
    return p.parse_args()


def test_clips(max_clips=None):
    """Test records in round-robin class order (so any prefix is class-balanced), each with a wrong-audio partner.

    The partner of clip i is the next clip in that order with a different class.
    """
    records = read_split("test")
    records = class_balanced_subset(records, max_clips or len(records))
    out = []
    for i, rec in enumerate(records):
        j = (i + 1) % len(records)
        while records[j]["category"] == rec["category"]:
            j = (j + 1) % len(records)
        out.append((rec, records[j]))
    return out


def load_frame_and_mask(rec, res, dilate_px):
    image = Image.open(frame_path(rec, 1)).convert("RGB").resize((res, res), Image.BICUBIC)
    mask = Image.open(mask_path(rec, 1)).convert("L").resize((res, res), Image.NEAREST)
    mask = mask.point(lambda v: 255 if v > 127 else 0)
    if dilate_px > 0:
        mask = mask.filter(ImageFilter.MaxFilter(2 * dilate_px + 1))
    return image, mask


# ---------------------------------------------------------------------------------------------- generate
def generate(args):
    from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline

    from models.audio_adapter import AudioAdapter
    from models.audio_token_conditioner import AudioTokenConditioner

    device = "cuda"
    out = Path(args.out)
    for d in ["gt", "mask"] + [f"images/{c}" for c in CONDITIONS]:
        (out / d).mkdir(parents=True, exist_ok=True)
    gen_args = {k: v for k, v in vars(args).items() if k not in ("command", "shard", "num_shards")}
    (out / "generate_args.json").write_text(json.dumps(gen_args, indent=2))

    ckpt = torch.load(args.ckpt, map_location=device)
    adapter = AudioAdapter(**ckpt["adapter_config"]).to(device).eval()
    adapter.load_state_dict(ckpt["adapter"])
    conditioner = AudioTokenConditioner(args.sd_path, template=ckpt["template"],
                                        num_tokens=ckpt["adapter_config"]["num_tokens"], device=device)
    tok = conditioner.tokenizer

    def encode_text(texts):
        ids = tok(texts, padding="max_length", max_length=tok.model_max_length, truncation=True,
                  return_tensors="pt").input_ids.to(device)
        return conditioner.text_encoder(input_ids=ids).last_hidden_state

    cache = torch.load(Path(CACHE_DIR) / "test.pt", map_location="cpu")
    if list(cache["layer_names"]) != list(ckpt["layer_names"]):
        raise RuntimeError(f"cached layers {cache['layer_names']} != checkpoint layers {ckpt['layer_names']}")
    row = {n: i for i, n in enumerate(cache["names"])}

    @torch.no_grad()
    def audio_embeds(names):
        idx = [row[n] for n in names]
        inter = [t[idx].float().to(device) for t in cache["intermediates"]]
        glob = cache["global_embed"][idx].float().to(device)
        return conditioner(adapter(inter, glob))

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.sd_path, dtype=torch.float16, variant="fp16", safety_checker=None,
        feature_extractor=None, requires_safety_checker=False).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    with torch.no_grad():
        uncond = encode_text([""])

    clips = list(enumerate(test_clips(args.max_clips)))[args.shard::args.num_shards]
    for start in range(0, len(clips), args.batch_size):
        batch = clips[start:start + args.batch_size]
        names = [rec["name"] for _, (rec, _) in batch]
        todo = [c for c in CONDITIONS if not all((out / "images" / c / f"{n}.png").is_file() for n in names)]
        if not todo:
            continue
        images, masks = zip(*[load_frame_and_mask(rec, args.resolution, args.dilate_px) for _, (rec, _) in batch])
        for n, im, m in zip(names, images, masks):
            im.save(out / "gt" / f"{n}.png")
            m.save(out / "mask" / f"{n}.png")
        with torch.no_grad():
            embeds = {
                "audio": audio_embeds(names),
                "shuffled": audio_embeds([partner["name"] for _, (_, partner) in batch]),
                "none": uncond.expand(len(batch), -1, -1),
                "text": encode_text([f"{ckpt['template']} {label_text(rec['category'])}" for _, (rec, _) in batch]),
            }
        n = len(batch)
        for cond in todo:
            gens = [torch.Generator(device).manual_seed(i) for i, _ in batch]  # seed = clip index, shared by conditions
            gen = pipe(prompt_embeds=embeds[cond].half(), negative_prompt_embeds=uncond.half().expand(n, -1, -1),
                       image=list(images), mask_image=list(masks), num_inference_steps=args.steps,
                       guidance_scale=args.guidance_scale, generator=gens).images
            for name, g, im, m in zip(names, gen, images, masks):
                Image.composite(g, im, m).save(out / "images" / cond / f"{name}.png")  # paste back the seen pixels
        print(f"shard {args.shard}: {start + n}/{len(clips)} clips", flush=True)


# ---------------------------------------------------------------------------------------------- report
def bbox(mask):
    ys, xs = np.nonzero(mask)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def generated_region(arr, mask):
    """Inpainted pixels only: crop to the mask's bounding box, gray out the rest, pad to a square."""
    y0, y1, x0, x1 = bbox(mask)
    a = arr.copy()
    a[~mask] = 127
    crop = a[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s, 3), 127, np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    return Image.fromarray(canvas)


def frechet_distance(a, b):
    from scipy import linalg

    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean.real))


class Scorer:
    def __init__(self, classes, device):
        import lpips
        from torchmetrics.functional.image import structural_similarity_index_measure
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.ssim_fn = structural_similarity_index_measure
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.clip = CLIPModel.from_pretrained(CLIP_PATH).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_PATH)
        text = self.proc(text=[f"a photo of {OBJECT_NOUNS[c]}." for c in classes], return_tensors="pt", padding=True)
        with torch.no_grad():
            t = self.clip.get_text_features(**text.to(device)).pooler_output  # projected embedding (transformers 5)
        self.text_feats = t / t.norm(dim=-1, keepdim=True)
        self.logit_scale = self.clip.logit_scale.exp()

    @torch.no_grad()
    def clip_image(self, images):
        """PIL images -> (L2-normalized features [N, 768], class probabilities [N, 23])."""
        pix = self.proc(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        f = self.clip.get_image_features(pixel_values=pix).pooler_output
        f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy(), (self.logit_scale * f @ self.text_feats.T).softmax(-1).cpu().numpy()

    @torch.no_grad()
    def pair(self, gen, gt, mask):
        """uint8 [H, W, 3] generated and ground-truth images, bool mask -> PSNR, SSIM, LPIPS."""
        diff = gen[mask].astype(np.float64) - gt[mask].astype(np.float64)
        psnr = 10 * np.log10(255.0 ** 2 / max(float((diff ** 2).mean()), 1e-10))
        y0, y1, x0, x1 = bbox(mask)

        def tensor(a, size=None):
            x = torch.from_numpy(np.ascontiguousarray(a[y0:y1, x0:x1])).permute(2, 0, 1)[None].float().to(self.device)
            if size:
                x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            return x / 255.0

        g, t = tensor(gen), tensor(gt)
        k = 11 if min(g.shape[-2:]) >= 11 else 3
        ssim = float(self.ssim_fn(g, t, data_range=1.0, kernel_size=k))
        lp = float(self.lpips(tensor(gen, 256) * 2 - 1, tensor(gt, 256) * 2 - 1).item())
        return psnr, ssim, lp


def score(args, classes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = Scorer(classes, device)
    out = Path(args.out)
    clips = [(i, rec, partner) for i, (rec, partner) in enumerate(test_clips(args.max_clips))
             if all((out / "images" / c / f"{rec['name']}.png").is_file() for c in CONDITIONS)]
    rows, feats = [], {c: [] for c in CONDITIONS + ["gt"]}
    for i, rec, partner in clips:
        name = rec["name"]
        gt = np.asarray(Image.open(out / "gt" / f"{name}.png").convert("RGB"))
        mask = np.asarray(Image.open(out / "mask" / f"{name}.png")) > 127
        if not mask.any():
            continue
        gens = {c: np.asarray(Image.open(out / "images" / c / f"{name}.png").convert("RGB")) for c in CONDITIONS}
        f, probs = scorer.clip_image([generated_region(gens[c], mask) for c in CONDITIONS]
                                     + [generated_region(gt, mask)])
        for k, c in enumerate(CONDITIONS + ["gt"]):
            feats[c].append(f[k])
        y_img, y_aud = classes.index(rec["category"]), classes.index(partner["category"])
        base = {"idx": i, "name": name, "category": rec["category"], "mask_frac": round(float(mask.mean()), 4),
                "gt_top1": int(probs[-1].argmax() == y_img)}
        for k, c in enumerate(CONDITIONS):
            psnr, ssim, lp = scorer.pair(gens[c], gt, mask)
            r = dict(base, condition=c, psnr=round(psnr, 3), ssim=round(ssim, 4), lpips=round(lp, 4),
                     top1=int(probs[k].argmax() == y_img), p_class=round(float(probs[k][y_img]), 4),
                     pred=classes[int(probs[k].argmax())])
            if c == "shuffled":
                r.update(audio_category=partner["category"], top1_audio_class=int(probs[k].argmax() == y_aud),
                         p_audio_class=round(float(probs[k][y_aud]), 4))
            rows.append(r)
        if len(feats["gt"]) % 100 == 0:
            print(f"scored {len(feats['gt'])}/{len(clips)}", flush=True)
    fd = {c: frechet_distance(np.stack(feats[c]), np.stack(feats["gt"])) for c in CONDITIONS}
    return rows, fd


def summarize(rows, fd, classes):
    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    def ci95(xs):
        xs = np.asarray(xs, dtype=np.float64)
        return float(1.96 * xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else float("nan")

    by = {c: [r for r in rows if r["condition"] == c] for c in CONDITIONS}
    overall = {}
    for c in CONDITIONS:
        rs = by[c]
        overall[c] = {k: mean([r[k] for r in rs]) for k in ("psnr", "ssim", "lpips", "top1", "p_class")}
        overall[c].update({f"{k}_ci95": ci95([r[k] for r in rs]) for k in ("psnr", "ssim", "lpips", "top1")})
        overall[c]["fd_clip"] = fd[c]
    overall["shuffled"]["top1_audio_class"] = mean([r["top1_audio_class"] for r in by["shuffled"]])
    overall["shuffled"]["p_audio_class"] = mean([r["p_audio_class"] for r in by["shuffled"]])
    gt_top1 = mean([r["gt_top1"] for r in by["audio"]])

    # paired comparison audio vs none per clip: does the right sound raise the CLIP probability of the class?
    none = {r["name"]: r for r in by["none"]}
    paired = {"p_class_audio_gt_none": mean([r["p_class"] > none[r["name"]]["p_class"] for r in by["audio"]]),
              "lpips_audio_lt_none": mean([r["lpips"] < none[r["name"]]["lpips"] for r in by["audio"]])}
    per_class = {cls: {c: mean([r["top1"] for r in by[c] if r["category"] == cls]) for c in CONDITIONS}
                 for cls in classes}
    for cls in classes:
        per_class[cls]["n"] = sum(r["category"] == cls for r in by["audio"])
    return {"num_clips": len(by["audio"]), "gt_clip_top1": gt_top1, "overall": overall, "paired": paired,
            "per_class": per_class}


def qualitative_grid(args, rows, out_path, n=8, t=160):
    """One row per clip (first n clips in class-balanced order, not cherry-picked)."""
    out = Path(args.out)
    by_name = {}
    for r in rows:
        by_name.setdefault(r["name"], {})[r["condition"]] = r
    names = [r["name"] for r in sorted(rows, key=lambda r: r["idx"]) if r["condition"] == "audio"][:n]
    cols = ["masked input", "ground truth", "own audio", "shuffled audio", "no audio", "text (oracle)"]
    sheet = Image.new("RGB", (len(cols) * t, 20 + len(names) * (t + 30)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, c in enumerate(cols):
        draw.text((k * t + 4, 4), c, fill="black")
    for j, name in enumerate(names):
        gt = Image.open(out / "gt" / f"{name}.png").convert("RGB")
        mask = Image.open(out / "mask" / f"{name}.png")
        masked = Image.composite(Image.new("RGB", gt.size, (128, 128, 128)), gt, mask)
        tiles = [masked, gt] + [Image.open(out / "images" / c / f"{name}.png") for c in CONDITIONS]
        rs = by_name[name]
        caps = [label_text(rs["audio"]["category"]), "",
                f"{rs['audio']['pred'][:16]}", f"{rs['shuffled']['pred'][:16]}",
                f"{rs['none']['pred'][:16]}", f"{rs['text']['pred'][:16]}"]
        y = 20 + j * (t + 30)
        for k, (im, cap) in enumerate(zip(tiles, caps)):
            sheet.paste(im.convert("RGB").resize((t, t), Image.BICUBIC), (k * t, y))
            draw.text((k * t + 3, y + t + 2), cap, fill="black")
        draw.text((3 * t + 3, y + t + 14), f"audio: {label_text(rs['shuffled']['audio_category'])[:20]}",
                  fill="gray")
    sheet.save(out_path)


def write_readme(args, s):
    out = Path(args.out)
    gen = json.loads((out / "generate_args.json").read_text())
    o = s["overall"]
    label = {"audio": "own audio (ours)", "shuffled": "wrong audio (shuffled)", "none": "no audio (SD-1.5 inpainting)",
             "text": "text prompt with class label (oracle)"}
    L = ["# Audio-conditioned inpainting: S4 test-set evaluation", "",
         "Generated by `scripts/eval_inpainting.py report`.", "",
         f"- **Data:** {s['num_clips']} AVSBench S4 test clips (all 23 classes), frame 1, ground-truth sounding-object "
         f"mask dilated by {gen['dilate_px']} px, 512 × 512.",
         f"- **Sampling:** SD-1.5 inpainting, DDIM {gen['steps']} steps, guidance {gen['guidance_scale']}, seed = clip "
         "index (identical across conditionings), generated pixels pasted back into the original frame.",
         "- **Checkpoint:** `runs/s4_v1/last.pt` (20k steps). Audio features: cached CLAP features of the test split.",
         "- PSNR on masked pixels; SSIM on the mask's bounding box; LPIPS (AlexNet) on the bounding box at 256 px; "
         "FD-CLIP = Fréchet distance of CLIP ViT-L/14 features of generated vs real region crops; CLIP top-1 = "
         "zero-shot accuracy over the 23 S4 classes on the generated region (outside the mask gray).",
         f"- CLIP top-1 on the **real** region (classifier ceiling): {100 * s['gt_clip_top1']:.1f}%.", "",
         "| conditioning | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FD-CLIP ↓ | CLIP top-1 ↑ | P(class) ↑ |",
         "|---|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        r = o[c]
        L.append(f"| {label[c]} | {r['psnr']:.2f} ± {r['psnr_ci95']:.2f} | {r['ssim']:.3f} | {r['lpips']:.3f} | "
                 f"{r['fd_clip']:.3f} | {100 * r['top1']:.1f}% ± {100 * r['top1_ci95']:.1f} | {r['p_class']:.3f} |")
    L += ["", "± is a 95% normal interval over clips.", "",
          f"- **Wrong audio follows the sound:** with another class's audio, CLIP assigns the generated region to "
          f"the *audio's* class {100 * o['shuffled']['top1_audio_class']:.1f}% of the time and to the image's class "
          f"{100 * o['shuffled']['top1']:.1f}% of the time.",
          f"- **Paired (same clip, seed, mask):** own audio gives a higher CLIP probability of the true class than no "
          f"audio for {100 * s['paired']['p_class_audio_gt_none']:.1f}% of clips, and a lower LPIPS for "
          f"{100 * s['paired']['lpips_audio_lt_none']:.1f}%.", "",
          "## Per class (CLIP top-1)", "",
          "| class | n | own audio | wrong audio | no audio | text (oracle) |", "|---|---|---|---|---|---|"]
    for cls, v in s["per_class"].items():
        L.append(f"| {label_text(cls)} | {v['n']} | {100 * v['audio']:.0f}% | {100 * v['shuffled']:.0f}% | "
                 f"{100 * v['none']:.0f}% | {100 * v['text']:.0f}% |")
    L += ["", "## Qualitative (first 8 clips in class-balanced order, not cherry-picked)", "",
          "![grid](grid.png)", "", "Caption under each generated image: CLIP top-1 class on the generated region.", ""]
    (out / "README.md").write_text("\n".join(L) + "\n")


def report(args):
    out = Path(args.out)
    classes = all_classes()
    rows, fd = score(args, classes)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    s = summarize(rows, fd, classes)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    qualitative_grid(args, rows, out / "grid.png")
    write_readme(args, s)
    o = s["overall"]
    print("  ".join(f"{c}: top1 {100 * o[c]['top1']:.1f}% lpips {o[c]['lpips']:.3f}" for c in CONDITIONS))
    print(f"report: {out / 'README.md'}")


def main():
    args = parse_args()
    if args.command == "generate":
        generate(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
