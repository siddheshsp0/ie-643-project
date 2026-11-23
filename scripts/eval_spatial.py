"""Does spatial audio put the right object in the right place? S4 test evaluation of four conditionings.

Clips: every S4 test clip whose sounding object is at most --max-width of the frame wide (173 clips in 21
classes at 0.33; most S4 objects are much larger and leave no "where" to choose). Frame 1 gets a hole 3x the
object's width and 1.6x its height with the object at one END of the hole (left end for even clips, right end
for odd ones), so a model that ignores location tends to draw it in the empty part. The clip's own audio is
placed at the object's true direction (the ray through its box centre, camera with a 60 deg field of view) and
distance (from its pixel height and the class size prior), and rendered as FOA in the synthetic room of
models/spatial_audio.py. Then, with the same seed and hole:

    none      no audio (empty prompt)
    mono      the Approach-1 model on the W channel: what one omnidirectional microphone records
    free      training-free spatial, masks: beam -> prompt, estimated direction and distance -> box;
              background pass over the hole, then the source inside box AND hole
    steer     training-free spatial, attention: one pass over the hole with the beam's prompt, its
              cross-attention steered to a heatmap at the same box (models/attention_steering.py)
    trained   the trained location model (runs/loc_v2b_multiscale): same prompt, heatmap at the same box, one pass

    generate  run the conditions (shardable; skips images that exist)
    report    OWLv2 detection of the clip's class inside the hole: found; centre hit = the detected box's
              centre lies inside the real object's box (right place, any size); IoU hit = IoU >= 0.3; centre
              error. CLIP top-1 on the hole, LPIPS on the hole's box, PSNR on the hole. Writes metrics.csv,
              summary.json, README.md and a grid

Usage (from the project root):
    CUDA_VISIBLE_DEVICES=1 python scripts/eval_spatial.py generate --conditions none mono free
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_spatial.py generate --conditions trained \
        --location-ckpt runs/loc_v2b_multiscale/last.pt --out outputs/spatial_eval_trained_v2b
    python scripts/eval_spatial.py report
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloaders.avs_s4 import AVSS4Dataset, label_text, mask_path, read_split, wav_path  # noqa: E402
from dataloaders.avs_s4_location import end_hole_box  # noqa: E402
from models.spatial_audio import (FOA_SR, direction_vector, estimate_distance, mic_to_camera,  # noqa: E402
                                  pixel_to_direction, project_point, synthesize_foa)
from models.spatial_pipeline import SIZE_PRIORS, SpatialAudioInpainter, make_box  # noqa: E402

CONDITIONS = ["none", "mono", "free", "steer", "trained"]
RES = 512
FOCAL = RES / 2 / math.tan(math.radians(30))  # 60 deg horizontal field of view
K = np.array([[FOCAL, 0, RES / 2], [0, FOCAL, RES / 2], [0, 0, 1.0]])
OBJECT_NOUNS = {  # same nouns as scripts/eval_inpainting.py
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
    p.add_argument("--out", default=str(ROOT / "outputs/spatial_eval"))
    p.add_argument("--ckpt", default=str(ROOT / "runs/s4_v1/last.pt"))
    p.add_argument("--location-ckpt", default=str(ROOT / "runs/loc_v2b_multiscale/last.pt"))
    p.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    p.add_argument("--max-width", type=float, default=0.33, help="largest object width (fraction of the frame)")
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--stride", type=int, default=1, help="every n-th selected clip (quick runs)")
    p.add_argument("--steer-floor", type=float, default=0.1)
    p.add_argument("--steer-boost", type=float, default=1.0)
    p.add_argument("--steer-active", type=float, default=0.5, help="fraction of steps that are steered")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    return p.parse_args()


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def select_clips(max_width, max_clips=None):
    """Test records whose frame-1 object (annotated mask) is at most max_width of the frame wide."""
    picked = []
    for rec in read_split("test"):
        m = np.asarray(Image.open(mask_path(rec, 1)).convert("L").resize((RES, RES), Image.NEAREST)) > 127
        if m.any():
            xs = np.nonzero(m)[1]
            if (xs.max() - xs.min() + 1) / RES <= max_width:
                picked.append(rec)
    return picked[:max_clips]


def true_source(category, bbox):
    """Where the clip's object is, as a sound source: direction of its box centre, distance from its height."""
    x0, y0, x1, y1 = bbox
    az, el = pixel_to_direction((x0 + x1) / 2, (y0 + y1) / 2, K)
    height_m = SIZE_PRIORS[category][0]
    # the synthetic room is indoor (critical distance 2 m): farther sources are capped at 4 m, the range in
    # which its distance and direction estimates are calibrated (a far helicopter would be 8+ m)
    r = float(np.clip(FOCAL * height_m / max(y1 - y0, 1), 0.5, 4.0))
    return az, el, r


def location_box(audio, cls, r_m):
    p = r_m * mic_to_camera(direction_vector(audio["az_deg"], audio["el_deg"]))
    u, v, z = project_point(K, np.eye(4), p)
    if not math.isfinite(u):  # estimated behind the camera: centre of the frame
        u, v, z = RES / 2, RES / 2, r_m
    height_m, aspect = SIZE_PRIORS[cls]
    h_px = FOCAL * height_m / max(z, 1e-3)
    return make_box(u, v, h_px, h_px * aspect, RES), (u, v)


def pil(x):
    return Image.fromarray(((x.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------------------------- generate
def generate(args):
    from models.location_unet import sample_location

    out = Path(args.out)
    for d in ["gt", "hole", "meta"] + [f"images/{c}" for c in args.conditions]:
        (out / d).mkdir(parents=True, exist_ok=True)
    ds = AVSS4Dataset("test", RES, train=False)
    records = select_clips(args.max_width, args.max_clips)
    model = SpatialAudioInpainter(args.ckpt, load_depth=False)
    if "trained" in args.conditions:
        model.load_location(args.location_ckpt)
    for i in list(range(len(records)))[::args.stride][args.shard::args.num_shards]:
        rec = records[i]
        name = rec["name"]
        todo = [c for c in args.conditions if not (out / "images" / c / f"{name}.png").is_file()]
        if not todo:
            continue
        image_t = ds._load_image(rec, 1)
        obj = ds._load_object_mask(rec, 1)[0].numpy() > 0.5
        image = pil(image_t)
        bbox = bbox_of(obj)
        hx0, hy0, hx1, hy1 = end_hole_box(bbox, i, RES)
        hole_np = obj.copy()
        hole_np[hy0:hy1, hx0:hx1] = True
        hole = Image.fromarray(hole_np.astype(np.uint8) * 255)
        az, el, r = true_source(rec["category"], bbox)
        mono, _ = librosa.load(str(wav_path(rec)), sr=FOA_SR, mono=True)
        foa = synthesize_foa(mono, az, el, r, seed=i)
        prompt, cls, p_cls, audio = model.listen(foa)
        r_est = estimate_distance(audio["diffuseness"])
        box, (u, v) = location_box(audio, cls, r_est)
        bx = np.zeros((RES, RES), bool)
        x0, y0, x1, y1 = (int(round(c)) for c in box)
        bx[y0:y1, x0:x1] = True
        location = Image.fromarray((bx & hole_np).astype(np.uint8) * 255)
        image.save(out / "gt" / f"{name}.png")
        hole.save(out / "hole" / f"{name}.png")
        meta = {"idx": i, "name": name, "category": rec["category"], "gt_bbox": bbox, "true_az": az, "true_el": el,
                "true_distance_m": r, "est_az": audio["az_deg"], "est_el": audio["el_deg"], "est_distance_m": r_est,
                "beam_class": cls, "beam_class_p": p_cls, "box": box, "u": u, "v": v}
        (out / "meta" / f"{name}.json").write_text(json.dumps(meta, default=float))
        for c in todo:
            if c == "none":
                im = model.fill(image, hole, hole, None, i, args.steps)
            elif c == "mono":
                prompt_w, _, _ = model.semantics(foa[0], FOA_SR)
                im = model.fill(image, hole, hole, prompt_w, i, args.steps)
            elif c == "steer":
                heat = model.heatmap(box, RES, model.device)[0, 0]
                im = model.fill_steered(image, hole, hole, prompt, heat, i, args.steps, floor=args.steer_floor,
                                        boost=args.steer_boost, active_frac=args.steer_active)
            elif c == "free":
                bg = model.fill(image, hole, hole, None, i, args.steps, negative=model.background_negative)
                im = model.fill(bg, location, location, prompt, i, args.steps, guidance_scale=9.0)
            else:
                mask = torch.from_numpy(hole_np.astype(np.float32))[None, None].to(model.device)
                heat = model.heatmap(box, RES, model.device)
                img_t = image_t[None].to(model.device)
                res = sample_location(model.location_unet, model.pipe.vae, model.pipe.scheduler.config, prompt,
                                      model.negative, img_t, mask, heat, args.steps, 7.5, seed=i)
                im = pil(res[0])
            im.save(out / "images" / c / f"{name}.png")
        print(f"[{i + 1}/{len(records)}] {name} {rec['category']}: az {az:+.0f}->{audio['az_deg']:+.0f}, "
              f"r {r:.1f}->{r_est:.1f} m, heard as {cls}", flush=True)


# ---------------------------------------------------------------------------------------------- report
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


class Scorer:
    def __init__(self, device):
        import lpips
        from transformers import CLIPModel, CLIPProcessor, Owlv2ForObjectDetection, Owlv2Processor

        self.device = device
        self.owl_proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()
        clip_path = ROOT / "models/clip-vit-large-patch14"
        self.clip = CLIPModel.from_pretrained(clip_path).to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(clip_path)
        self.classes = sorted(OBJECT_NOUNS)
        text = self.clip_proc(text=[f"a photo of {OBJECT_NOUNS[c]}." for c in self.classes], return_tensors="pt",
                              padding=True)
        with torch.no_grad():
            t = self.clip.get_text_features(**text.to(device)).pooler_output
        self.text = t / t.norm(dim=-1, keepdim=True)
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def detect(self, image, category, hole):
        """Best box of the class whose centre is inside the hole: (box, score) or (None, 0)."""
        inp = self.owl_proc(text=[[f"a photo of {OBJECT_NOUNS[category]}"]], images=image,
                            return_tensors="pt").to(self.device)
        r = self.owl_proc.post_process_grounded_object_detection(
            outputs=self.owl(**inp), target_sizes=torch.tensor([image.size[::-1]]), threshold=0.1)[0]
        best, score = None, 0.0
        for b, s in zip(r["boxes"].tolist(), r["scores"].tolist()):
            cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
            if 0 <= cx < RES and 0 <= cy < RES and hole[cy, cx] and s > score:
                best, score = b, s
        return best, score

    @torch.no_grad()
    def clip_top1(self, gen, hole, category):
        a = np.asarray(gen).copy()
        a[~hole] = 127
        x0, y0, x1, y1 = bbox_of(hole)
        pix = self.clip_proc(images=[Image.fromarray(a[y0:y1, x0:x1])], return_tensors="pt")["pixel_values"]
        f = self.clip.get_image_features(pixel_values=pix.to(self.device)).pooler_output
        f = f / f.norm(dim=-1, keepdim=True)
        return int(self.classes[int((f @ self.text.T).argmax())] == category)

    @torch.no_grad()
    def fidelity(self, gen, gt, hole):
        g, t = np.asarray(gen).astype(np.float64), np.asarray(gt).astype(np.float64)
        psnr = 10 * np.log10(255 ** 2 / max(((g[hole] - t[hole]) ** 2).mean(), 1e-10))
        x0, y0, x1, y1 = bbox_of(hole)

        def ten(a):
            x = torch.from_numpy(np.ascontiguousarray(a[y0:y1, x0:x1])).permute(2, 0, 1)[None].float() / 127.5 - 1
            return torch.nn.functional.interpolate(x, size=(256, 256), mode="bilinear").to(self.device)

        return psnr, float(self.lpips(ten(g), ten(t)).item())


def report(args):
    out = Path(args.out)
    scorer = Scorer("cuda" if torch.cuda.is_available() else "cpu")
    conds = [c for c in CONDITIONS if (out / "images" / c).is_dir()]
    rows = []
    metas = sorted((json.loads(p.read_text()) for p in (out / "meta").glob("*.json")), key=lambda m: m["idx"])
    for m in metas:
        name = m["name"]
        if not all((out / "images" / c / f"{name}.png").is_file() for c in conds):
            continue
        gt = Image.open(out / "gt" / f"{name}.png").convert("RGB")
        hole = np.asarray(Image.open(out / "hole" / f"{name}.png")) > 127
        doa_err = math.degrees(math.acos(np.clip(np.dot(direction_vector(m["true_az"], m["true_el"]),
                                                        direction_vector(m["est_az"], m["est_el"])), -1, 1)))
        for c in conds:
            gen = Image.open(out / "images" / c / f"{name}.png").convert("RGB")
            box, score = scorer.detect(gen, m["category"], hole)
            gb = m["gt_bbox"]
            cerr = (math.hypot((box[0] + box[2] - gb[0] - gb[2]) / 2, (box[1] + box[3] - gb[1] - gb[3]) / 2) / RES
                    if box else float("nan"))
            psnr, lp = scorer.fidelity(gen, gt, hole)
            centre_in = box is not None and gb[0] <= (box[0] + box[2]) / 2 <= gb[2] and \
                gb[1] <= (box[1] + box[3]) / 2 <= gb[3]
            rows.append({"name": name, "category": m["category"], "condition": c, "found": int(box is not None),
                         "hit": int(centre_in), "iou_hit": int(box is not None and iou(box, gb) >= 0.3),
                         "iou": iou(box, gb) if box else 0.0,
                         "centre_err": cerr, "det_score": score, "clip_top1": scorer.clip_top1(gen, hole, m["category"]),
                         "psnr": psnr, "lpips": lp, "doa_err_deg": doa_err,
                         "dist_err": abs(m["est_distance_m"] - m["true_distance_m"]) / m["true_distance_m"],
                         "beam_class_ok": int(m["beam_class"] == m["category"])})
        print(f"scored {name}", flush=True)
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def mean(xs):
        xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
        return float(np.mean(xs)) if xs else float("nan")

    def ci(xs):
        return float(1.96 * np.std(xs, ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else float("nan")

    summary = {"num_clips": len(rows) // len(conds), "conditions": {}}
    for c in conds:
        rs = [r for r in rows if r["condition"] == c]
        summary["conditions"][c] = {k: mean([r[k] for r in rs]) for k in
                                    ("found", "hit", "iou_hit", "iou", "centre_err", "clip_top1", "psnr",
                                     "lpips")}
        summary["conditions"][c]["hit_ci95"] = ci([r["hit"] for r in rs])
    first = [r for r in rows if r["condition"] == conds[0]]
    summary["audio"] = {"doa_err_deg_mean": mean([r["doa_err_deg"] for r in first]),
                        "doa_err_deg_median": float(np.median([r["doa_err_deg"] for r in first])),
                        "dist_err_mean": mean([r["dist_err"] for r in first]),
                        "beam_class_acc": mean([r["beam_class_ok"] for r in first])}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_readme(out, summary, conds)
    scored = [m for m in metas if all((out / "images" / c / f"{m['name']}.png").is_file() for c in conds)]
    grid(out, conds, scored[:8])
    for c, s in summary["conditions"].items():
        print(f"{c:8s} found {100 * s['found']:5.1f}%  centre hit {100 * s['hit']:5.1f}%  IoU hit "
              f"{100 * s['iou_hit']:5.1f}%  CLIP {100 * s['clip_top1']:5.1f}%  "
              f"LPIPS {s['lpips']:.3f}  PSNR {s['psnr']:.2f}")


LABELS = {"none": "no audio", "mono": "mono audio (W channel)", "free": "spatial, training-free (masks)",
          "steer": "spatial, training-free (attention)", "trained": "spatial, trained location"}


def write_readme(out, s, conds):
    a = s["audio"]
    L = ["# Spatial audio: right object, right place (S4 test)", "",
         "Generated by `scripts/eval_spatial.py report`.", "",
         f"- **Data:** {s['num_clips']} AVSBench S4 test clips whose sounding object is at most a third of the frame "
         "wide, frame 1, hole 3x the object's width and 1.6x its height with the object at one end of the hole "
         "(alternating left and right). The clip's own audio is rendered as FOA at the object's true direction "
         "(60 deg camera) and "
         "distance (from its pixel height and the class size prior), in the synthetic room of "
         "`models/spatial_audio.py` (critical distance 2 m, RT60 0.5 s).",
         f"- **Audio analysis:** direction error {a['doa_err_deg_mean']:.1f} deg mean ({a['doa_err_deg_median']:.1f} "
         f"median); distance error {100 * a['dist_err_mean']:.0f}% mean; CLAP zero-shot class from the beam "
         f"{100 * a['beam_class_acc']:.1f}%.",
         "- **Found:** OWLv2 detects the clip's class inside the hole (score >= 0.1). **Centre hit:** the detected "
         "box's centre lies inside the real object's box, i.e. the right object in the right place (any size). "
         "**IoU hit:** IoU >= 0.3 with the real box (place and size). CLIP top-1 on the hole; LPIPS / PSNR against "
         "the real frame. Sources are rendered at 0.5-4 m (the indoor room model's range).", "",
         "| conditioning | found | centre hit (right place) | IoU hit | centre error (frac. of width) | CLIP top-1 "
         "| LPIPS | PSNR |", "|---|---|---|---|---|---|---|---|"]
    for c in conds:
        r = s["conditions"][c]
        L.append(f"| {LABELS[c]} | {100 * r['found']:.1f}% | **{100 * r['hit']:.1f}%** ± {100 * r['hit_ci95']:.1f} | "
                 f"{100 * r['iou_hit']:.1f}% | {r['centre_err']:.3f} | {100 * r['clip_top1']:.1f}% | {r['lpips']:.3f} | "
                 f"{r['psnr']:.2f} |")
    L += ["", "![grid](grid.png)", "", "Grid: the first 8 clips (not cherry-picked). Columns: masked input, real frame, "
          "then each conditioning; the red box is where the spatial methods were told the source is.", ""]
    (out / "README.md").write_text("\n".join(L) + "\n")


def grid(out, conds, metas, t=192):
    cols = ["masked input", "real"] + [LABELS[c] for c in conds]
    sheet = Image.new("RGB", (len(cols) * t, 22 + len(metas) * (t + 18)), "white")
    d = ImageDraw.Draw(sheet)
    for k, c in enumerate(cols):
        d.text((k * t + 3, 4), c, fill="black")
    for j, m in enumerate(metas):
        name = m["name"]
        gt = Image.open(out / "gt" / f"{name}.png").convert("RGB")
        hole = Image.open(out / "hole" / f"{name}.png")
        masked = Image.composite(Image.new("RGB", gt.size, (128, 128, 128)), gt, hole)
        tiles = [masked, gt] + [Image.open(out / "images" / c / f"{name}.png").convert("RGB") for c in conds]
        y = 22 + j * (t + 18)
        for k, im in enumerate(tiles):
            im = im.copy()
            if k >= 2 and conds[k - 2] in ("free", "steer", "trained"):
                ImageDraw.Draw(im).rectangle(m["box"], outline=(227, 73, 72), width=3)
            sheet.paste(im.resize((t, t), Image.BICUBIC), (k * t, y))
        d.text((3, y + t + 2), f"{label_text(m['category'])}, az {m['true_az']:+.0f} deg", fill="black")
    sheet.save(out / "grid.png")


def main():
    args = parse_args()
    generate(args) if args.command == "generate" else report(args)


if __name__ == "__main__":
    main()
