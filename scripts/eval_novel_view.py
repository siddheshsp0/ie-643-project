"""Evaluation grid for the look-around pipeline: 8 S4 test scenes x poses x {no audio, 4 sounds} x seeds.

Audio and no-audio images share seeds and masks. `generate` runs the pipeline (shardable over GPUs, skips
existing images); `report` scores only the generated pixels with CLIP zero-shot over the 23 S4 classes and
writes metrics.csv, summary.json, contact sheets and README.md.

Usage (from the project root):
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_novel_view.py generate --shard 0 --num-shards 2 &
    CUDA_VISIBLE_DEVICES=1 python scripts/eval_novel_view.py generate --shard 1 --num-shards 2
    python scripts/eval_novel_view.py report

Pose variants (pass report the same --out and --yaws):
    --pose no-forward     tz = |sin(yaw)|, so negative yaw moves back instead of forward
    --pose rotation-only  tx = tz = 0, no parallax
    --focal-256 200       focal length at 256 px (MegaScenes uses 80, about 116 deg FoV; 200 is about 65 deg)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloaders.avs_s4 import all_classes, frame_path, wav_path  # noqa: E402
from models.megascenes_warp import MEGASCENES_FOCAL_256, orbit_extrinsic  # noqa: E402
from models.novel_view_pipeline import center_crop_square, colorize_disparity, warp_view  # noqa: E402

SCENES = [  # (scene id, S4 category, clip id); frame 1 of the test clip is the input view
    ("desk", "typing_on_computer_keyboard", "-6AOy8GAMpM"),
    ("doorway_dog", "dog_barking", "-AWooU1PDiM"),
    ("fireplace_room", "playing_violin", "8TZcN7vYW8E"),
    ("piano_room", "playing_violin", "4Cc4hviIWts"),
    ("bathroom_door", "baby_laughter", "iXKrKQ-JZVo"),
    ("bookshelf_room", "baby_laughter", "mINZOofPPdA"),
    ("country_path", "horse_clip-clop", "6GD0dWvfDc4"),
    ("bus_street", "driving_buses", "-II-7qJxKPc"),
]
AUDIOS = {  # condition -> (S4 category, clip id) of the conditioning WAV
    "cat": ("cat_meowing", "0UN02RkbG_U"),
    "dog": ("dog_barking", "-AWooU1PDiM"),
    "baby": ("baby_laughter", "MrIfjRInMLU"),
    "guitar": ("playing_acoustic_guitar", "-3L1rzGAD_o"),
}
CONDITIONS = ["none"] + list(AUDIOS)
OBJECT_NOUNS = {  # CLIP prompt "a photo of {noun}." per S4 class
    "ambulance_siren": "an ambulance", "baby_laughter": "a baby", "cap_gun_shooting": "a toy gun",
    "cat_meowing": "a cat", "chainsawing_trees": "a chainsaw", "coyote_howling": "a coyote",
    "dog_barking": "a dog", "driving_buses": "a bus", "female_singing": "a woman singing",
    "helicopter": "a helicopter", "horse_clip-clop": "a horse", "lawn_mowing": "a lawn mower",
    "lions_roaring": "a lion", "male_speech": "a man speaking", "mynah_bird_singing": "a bird",
    "playing_acoustic_guitar": "an acoustic guitar", "playing_glockenspiel": "a glockenspiel",
    "playing_piano": "a piano", "playing_tabla": "tabla drums", "playing_ukulele": "a ukulele",
    "playing_violin": "a violin", "race_car": "a race car", "typing_on_computer_keyboard": "a computer keyboard",
}
CLIP_PATH = ROOT / "models/clip-vit-large-patch14"
POSES = {  # --pose -> README description
    "orbit": "MegaScenes orbit",
    "no-forward": "MegaScenes orbit, camera moves back for both directions (tz = |sin yaw|)",
    "rotation-only": "rotation only, no translation",
}


def s4_record(category, name):
    return {"split": "test", "category": category, "name": name}


def pose_extrinsic(yaw, pose):
    if pose == "rotation-only":
        return orbit_extrinsic(yaw, tx=0.0, tz=0.0)
    if pose == "no-forward":
        return orbit_extrinsic(yaw, tz=abs(np.sin(np.radians(yaw))))
    return orbit_extrinsic(yaw)


def fov_deg(focal_256):
    return float(np.degrees(2 * np.arctan(128.0 / focal_256)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["generate", "report"])
    p.add_argument("--out", default=str(ROOT / "outputs/novel_view_eval/v1"))
    p.add_argument("--ckpt", default=str(ROOT / "runs/s4_v1/last.pt"), help="audio adapter checkpoint")
    p.add_argument("--yaws", type=float, nargs="+", default=[15.0, -15.0])
    p.add_argument("--pose", choices=list(POSES), default="orbit", help="translation schedule for every yaw")
    p.add_argument("--focal-256", type=float, default=MEGASCENES_FOCAL_256,
                   help="focal length in px for a 256 px image (MegaScenes uses 80)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--dilate-px", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def view_dir(out, scene_id, yaw):
    return Path(out) / "scenes" / scene_id / f"yaw{yaw:+g}"


def frac(mask):
    return float((np.asarray(mask) > 127).mean())


def generate(args):
    from models.novel_view_pipeline import NovelViewAudioInpainter

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gen_args = {k: v for k, v in vars(args).items() if k not in ("command", "shard", "num_shards", "overwrite")}
    (out / "generate_args.json").write_text(json.dumps(dict(gen_args, scenes=SCENES, audios=AUDIOS), indent=2))

    model = NovelViewAudioInpainter(args.ckpt)
    embeds = {c: model.audio_embeds(wav_path(s4_record(*AUDIOS[c]))) for c in AUDIOS}
    embeds["none"] = None
    res = args.resolution
    for scene_id, category, name in SCENES[args.shard::args.num_shards]:
        sdir = out / "scenes" / scene_id
        sdir.mkdir(parents=True, exist_ok=True)
        image = center_crop_square(Image.open(frame_path(s4_record(category, name), 1)).convert("RGB"))
        src = image.resize((res, res), Image.LANCZOS)
        disparity, depth = model.depth(image, res)
        src.save(sdir / "input.png")
        colorize_disparity(disparity).save(sdir / "depth.png")
        np.save(sdir / "depth.npy", depth)

        stats = {}
        for yaw in args.yaws:
            vdir = view_dir(out, scene_id, yaw)
            vdir.mkdir(parents=True, exist_ok=True)
            E = pose_extrinsic(yaw, args.pose)
            warped, hole, mask = warp_view(src, depth, E, args.focal_256, dilate_px=args.dilate_px)
            ms_warped, ms_hole, _ = warp_view(src, depth, E, args.focal_256, prune="megascenes", dilate_px=0)
            for fname, im in [("warped", warped), ("mask", hole), ("inpaint_mask", mask),
                              ("megascenes_warped", ms_warped), ("megascenes_mask", ms_hole)]:
                im.save(vdir / f"{fname}.png")
            stats[f"{yaw:+g}"] = {"translation": E[:3, 3].round(4).tolist(), "hole_frac": frac(hole),
                                  "inpaint_frac": frac(mask), "megascenes_hole_frac": frac(ms_hole)}

            for cond in CONDITIONS:
                paths = [vdir / cond / f"seed{s}.png" for s in args.seeds]
                if not args.overwrite and all(p.is_file() for p in paths):
                    continue
                images = model.inpaint(warped, hole, mask, embeds[cond], seeds=args.seeds, steps=args.steps,
                                       guidance_scale=args.guidance_scale, feather_px=max(args.dilate_px / 2, 1))
                paths[0].parent.mkdir(exist_ok=True)
                for p, im in zip(paths, images):
                    im.save(p)
                print(f"{scene_id} yaw {yaw:+g} {cond}: {len(images)} images", flush=True)
        (sdir / "scene.json").write_text(json.dumps(
            {"scene": scene_id, "category": category, "clip": name, "views": stats}, indent=2))


def generated_region(image, mask):
    """Inpainted pixels only: crop to the mask's bounding box, gray out the rest, pad to a square."""
    m = np.asarray(mask) > 127
    ys, xs = np.nonzero(m)
    arr = np.asarray(image.convert("RGB")).copy()
    arr[~m] = 127
    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s, 3), 127, np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    return Image.fromarray(canvas)


class ClipZeroShot:
    def __init__(self, classes, device):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP_PATH).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_PATH)
        prompts = [f"a photo of {OBJECT_NOUNS[c]}." for c in classes]
        self.text = self.processor(text=prompts, return_tensors="pt", padding=True).to(device)

    @torch.no_grad()
    def __call__(self, images):
        pix = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        logits = self.model(**self.text, pixel_values=pix).logits_per_image
        return logits.softmax(-1).cpu().numpy()


def score(args, classes):
    """One metrics row per generated image."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip = ClipZeroShot(classes, device)
    targets = {c: classes.index(AUDIOS[c][0]) for c in AUDIOS}
    rows = []
    for scene_id, _, _ in SCENES:
        for yaw in args.yaws:
            vdir = view_dir(args.out, scene_id, yaw)
            mask = Image.open(vdir / "inpaint_mask.png")
            for cond in CONDITIONS:
                paths = [vdir / cond / f"seed{s}.png" for s in args.seeds]
                probs = clip([generated_region(Image.open(p), mask) for p in paths])
                for seed, p in zip(args.seeds, probs):
                    top = int(p.argmax())
                    row = {"scene": scene_id, "yaw": f"{yaw:+g}", "condition": cond, "seed": seed,
                           "inpaint_frac": round(frac(mask), 4), "top1": classes[top], "top1_p": round(float(p[top]), 4)}
                    row.update({f"p_{c}": round(float(p[i]), 4) for c, i in targets.items()})
                    rows.append(row)
    return rows


def summarize(rows):
    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    none = {(r["scene"], r["yaw"], r["seed"]): r for r in rows if r["condition"] == "none"}
    per_audio = {}
    for c in AUDIOS:
        target = AUDIOS[c][0]
        audio = [r for r in rows if r["condition"] == c]
        per_audio[c] = {
            "target_class": target,
            "top1_audio": mean([r["top1"] == target for r in audio]),
            "top1_no_audio": mean([r["top1"] == target for r in none.values()]),
            "p_target_audio": mean([r[f"p_{c}"] for r in audio]),
            "p_target_no_audio": mean([r[f"p_{c}"] for r in none.values()]),
            "win_rate": mean([r[f"p_{c}"] > none[(r["scene"], r["yaw"], r["seed"])][f"p_{c}"] for r in audio]),
            "by_yaw": {y: {"top1_audio": mean([r["top1"] == target for r in audio if r["yaw"] == y]),
                           "top1_no_audio": mean([r["top1"] == target for r in none.values() if r["yaw"] == y])}
                       for y in sorted({r["yaw"] for r in rows})},
        }
    matrix = {cond: {c: mean([r[f"p_{c}"] for r in rows if r["condition"] == cond]) for c in AUDIOS}
              for cond in CONDITIONS}
    per_scene = {}
    for scene_id, _, _ in SCENES:
        audio = [r for r in rows if r["scene"] == scene_id and r["condition"] in AUDIOS]
        base = [r for r in rows if r["scene"] == scene_id and r["condition"] == "none"]
        per_scene[scene_id] = {
            "top1_audio": mean([r["top1"] == AUDIOS[r["condition"]][0] for r in audio]),
            # how often a no-audio image already shows a sound's class, averaged over the 4 sounds
            "top1_no_audio": mean([r["top1"] == AUDIOS[c][0] for r in base for c in AUDIOS]),
        }
    return {"per_audio": per_audio, "prob_matrix": matrix, "per_scene": per_scene,
            "overall": {"top1_audio": mean([per_audio[c]["top1_audio"] for c in AUDIOS]),
                        "top1_no_audio": mean([per_audio[c]["top1_no_audio"] for c in AUDIOS]),
                        "win_rate": mean([per_audio[c]["win_rate"] for c in AUDIOS])}}


def tile(im, size, caption=None):
    canvas = Image.new("RGB", (size, size + (16 if caption else 0)), "white")
    canvas.paste(im.convert("RGB").resize((size, size), Image.BICUBIC), (0, 0))
    if caption:
        ImageDraw.Draw(canvas).text((3, size + 2), caption, fill="black")
    return canvas


def scene_sheet(args, scene_id, rows, t=192, label_w=90):
    """Contact sheet for one scene: header with input/depth/warps, then one row per condition."""
    sdir = Path(args.out) / "scenes" / scene_id
    meta = json.loads((sdir / "scene.json").read_text())
    by_key = {(r["yaw"], r["condition"], r["seed"]): r for r in rows if r["scene"] == scene_id}
    header = [("input", Image.open(sdir / "input.png")), ("depth (DA-V2)", Image.open(sdir / "depth.png"))]
    for yaw in args.yaws:
        vdir = view_dir(args.out, scene_id, yaw)
        hole = meta["views"][f"{yaw:+g}"]["hole_frac"]
        header += [(f"warped {yaw:+g} deg", Image.open(vdir / "warped.png")),
                   (f"unseen {100 * hole:.0f}%", Image.open(vdir / "mask.png"))]
    cols = max(len(header), len(args.yaws) * len(args.seeds))
    row_h = t + 16 + 18
    sheet = Image.new("RGB", (label_w + cols * t, 24 + row_h * (1 + len(CONDITIONS))), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), f"{scene_id}  (S4 test {meta['category']}/{meta['clip']})   rows: conditioning sound; "
                      f"columns: seeds {args.seeds} for yaw {' / '.join(f'{y:+g}' for y in args.yaws)}; "
                      f"caption: CLIP top-1 on the generated pixels", fill="black")
    for i, (cap, im) in enumerate(header):
        draw.text((label_w + i * t + 3, 24), cap, fill="black")
        sheet.paste(tile(im, t), (label_w + i * t, 24 + 18))
    for j, cond in enumerate(CONDITIONS):
        y = 24 + row_h * (j + 1)
        draw.text((4, y + 18 + t // 2), "no audio" if cond == "none" else f"{cond} audio", fill="black")
        for a, yaw in enumerate(args.yaws):
            for b, seed in enumerate(args.seeds):
                r = by_key[(f"{yaw:+g}", cond, seed)]
                x = label_w + (a * len(args.seeds) + b) * t
                if b == 0:
                    draw.text((x + 3, y), f"yaw {yaw:+g}", fill="black")
                cap = f"{OBJECT_NOUNS[r['top1']].split(' ', 1)[-1][:18]} {r['top1_p']:.2f}"
                img = Image.open(view_dir(args.out, scene_id, yaw) / cond / f"seed{seed}.png")
                sheet.paste(tile(img, t, cap), (x, y + 18))
    return sheet


def hole_rule_sheet(args, t=192):
    """Per scene at the first yaw: input, MegaScenes angle rule, our stretch rule."""
    yaw = args.yaws[0]
    sheet = Image.new("RGB", (3 * t + 120, 24 + len(SCENES) * (t + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), f"warp to yaw {yaw:+g} deg: input | MegaScenes hole rule (angle > 70 deg) | stretch rule (used)",
              fill="black")
    for i, (scene_id, _, _) in enumerate(SCENES):
        sdir = Path(args.out) / "scenes" / scene_id
        vdir = view_dir(args.out, scene_id, yaw)
        v = json.loads((sdir / "scene.json").read_text())["views"][f"{yaw:+g}"]
        y = 24 + i * (t + 18)
        draw.text((4, y + t // 2), scene_id, fill="black")
        for k, (cap, path) in enumerate([("input", sdir / "input.png"),
                                         (f"megascenes: {100 * v['megascenes_hole_frac']:.0f}% missing",
                                          vdir / "megascenes_warped.png"),
                                         (f"stretch: {100 * v['hole_frac']:.0f}% missing", vdir / "warped.png")]):
            sheet.paste(tile(Image.open(path), t, cap), (120 + k * t, y))
    return sheet


def pct(x):
    return f"{100 * x:.0f}%"


def write_readme(args, summary, scenes_meta):
    out = Path(args.out)
    gen = json.loads((out / "generate_args.json").read_text())
    pose, focal = gen.get("pose", "orbit"), gen.get("focal_256", MEGASCENES_FOCAL_256)
    n_img = len(SCENES) * len(args.yaws) * len(CONDITIONS) * len(args.seeds)
    L = ["# Look-around evaluation (depth → MegaScenes warp → audio-conditioned inpainting)", "",
         "Generated by `scripts/eval_novel_view.py report`.", "",
         f"- **Grid:** {len(SCENES)} scenes × {len(args.yaws)} poses (yaw {', '.join(f'{y:+g}°' for y in args.yaws)}, "
         f"{POSES[pose]}; focal {focal:g} px at 256 px, ≈{fov_deg(focal):.0f}° FoV) × "
         f"{len(CONDITIONS)} conditions (no audio + {', '.join(AUDIOS)}) × "
         f"{len(args.seeds)} seeds = **{n_img} images**, {args.steps} DDIM steps, guidance {args.guidance_scale}, "
         f"mask dilation {args.dilate_px} px.",
         "- Scenes and sounds are AVSBench S4 **test** clips (frames are 224 px, upscaled to 512).",
         "- Audio and no-audio images use the same seeds and masks, so they differ only in the conditioning.",
         "- **Score:** CLIP ViT-L/14 zero-shot over the 23 S4 classes (`a photo of {object}.`), shown **only the "
         "generated pixels** (mask bounding box, everything else gray). It measures whether the requested object "
         "appears in the revealed region; it does not judge realism.", "",
         "## Does the sound decide what appears?", "",
         "| sound | CLIP class | top-1 with audio | top-1 without audio | P(class) with / without | audio > no-audio (paired) |",
         "|---|---|---|---|---|---|"]
    for c, s in summary["per_audio"].items():
        L.append(f"| {c} | {OBJECT_NOUNS[s['target_class']]} | **{pct(s['top1_audio'])}** | {pct(s['top1_no_audio'])} | "
                 f"{s['p_target_audio']:.2f} / {s['p_target_no_audio']:.2f} | {pct(s['win_rate'])} |")
    o = summary["overall"]
    L += [f"| **all** | | **{pct(o['top1_audio'])}** | {pct(o['top1_no_audio'])} | | {pct(o['win_rate'])} |", "",
          "Each audio row has " f"{len(SCENES) * len(args.yaws) * len(args.seeds)} images; "
          "\"without audio\" scores the same no-audio images against that row's class.", "",
          "**Mean CLIP probability of each class (columns) per conditioning (rows):** the diagonal should dominate.", "",
          "| conditioning | " + " | ".join(OBJECT_NOUNS[AUDIOS[c][0]] for c in AUDIOS) + " |",
          "|---|" + "---|" * len(AUDIOS)]
    for cond, ps in summary["prob_matrix"].items():
        cells = [f"**{ps[c]:.2f}**" if c == cond else f"{ps[c]:.2f}" for c in AUDIOS]
        L.append(f"| {'no audio' if cond == 'none' else cond} | " + " | ".join(cells) + " |")
    L += ["", "**By pose** (top-1 hit rate, audio vs no audio):", "",
          "| sound | " + " | ".join(f"yaw {y:+g}°" for y in args.yaws) + " |", "|---|" + "---|" * len(args.yaws)]
    for c, s in summary["per_audio"].items():
        L.append(f"| {c} | " + " | ".join(f"{pct(v['top1_audio'])} vs {pct(v['top1_no_audio'])}"
                                         for v in s["by_yaw"].values()) + " |")
    L += ["", "## Per scene", "",
          "| scene | unseen pixels " + " / ".join(f"{y:+g}°" for y in args.yaws)
          + " | inpainted (dilated) | top-1 hit, audio | top-1 hit, no audio | sheet |", "|---|---|---|---|---|---|"]
    for scene_id, _, _ in SCENES:
        views = scenes_meta[scene_id]["views"]
        s = summary["per_scene"][scene_id]
        L.append(f"| {scene_id} | " + " / ".join(pct(views[f'{y:+g}']['hole_frac']) for y in args.yaws) + " | "
                 + " / ".join(pct(views[f'{y:+g}']['inpaint_frac']) for y in args.yaws)
                 + f" | {pct(s['top1_audio'])} | {pct(s['top1_no_audio'])} | [png](sheets/{scene_id}.png) |")
    L += ["", "## Contact sheets", "",
          "Header: input, depth, and for each pose the warped view (black = never seen) and the unseen-pixel mask. "
          "Rows: conditioning sound. Columns: seeds per pose. Caption: CLIP top-1 class and probability on the "
          "generated pixels.", ""]
    L += [f"### {scene_id}\n\n![{scene_id}](sheets/{scene_id}.png)\n" for scene_id, _, _ in SCENES]
    L += ["## Hole rule: MegaScenes angle rule vs stretch rule", "",
          "![hole rules](sheets/hole_rules.png)", "",
          "## Files", "",
          "```",
          "scenes/<scene>/input.png, depth.png, depth.npy, scene.json (pose, unseen/inpainted fractions)",
          "scenes/<scene>/yaw±15/warped.png, mask.png (unseen), inpaint_mask.png (dilated),",
          "                      megascenes_warped.png, megascenes_mask.png (MegaScenes hole rule, for comparison)",
          "scenes/<scene>/yaw±15/<none|cat|dog|baby|guitar>/seed<k>.png   final images",
          "sheets/<scene>.png, sheets/hole_rules.png",
          "metrics.csv (one row per image), summary.json, generate_args.json",
          "```"]
    (out / "README.md").write_text("\n".join(L) + "\n")


def report(args):
    out = Path(args.out)
    classes = all_classes()
    rows = score(args, classes)
    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    (out / "sheets").mkdir(exist_ok=True)
    scenes_meta = {}
    for scene_id, _, _ in SCENES:
        scenes_meta[scene_id] = json.loads((out / "scenes" / scene_id / "scene.json").read_text())
        scene_sheet(args, scene_id, rows).save(out / "sheets" / f"{scene_id}.png")
    hole_rule_sheet(args).save(out / "sheets" / "hole_rules.png")
    write_readme(args, summary, scenes_meta)
    o = summary["overall"]
    print(f"top-1 hit rate: audio {pct(o['top1_audio'])} vs no audio {pct(o['top1_no_audio'])}; "
          f"paired win rate {pct(o['win_rate'])}")
    print(f"report: {out / 'README.md'}")


def main():
    args = parse_args()
    if args.command == "generate":
        generate(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
