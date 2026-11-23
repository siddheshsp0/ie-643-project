"""Around the corner: does the generated object appear where the sound comes from?

Each scene (8 S4 test frames of the look-around evaluation, 5 unseen Kaggle photos, our sofa photo) gets a
sound (cat, dog, baby, guitar; S4 test clips) placed OUTSIDE the field of view, 65 deg to the left or to the
right, 2.5 m away, rendered as FOA in the synthetic room. Then:

    mono      the look-around pipeline of the intensive assessment: a fixed +20 deg camera turn (a mono
              recording cannot say which side), the whole unseen region filled with the W channel's prompt
    spatial   the training-free spatial pipeline: the camera turns toward the estimated direction, and the
              unseen region is filled with the beam's prompt, its attention steered to the projected source
    trained   same camera turn and heatmap, filled by the trained location model (--location-ckpt)

Score: OWLv2 detects the sound's class in the generated (unseen) pixels; hit = the detected box contains the
true source direction projected into that method's own view (so turning the wrong way can never hit).

Usage (from the project root):
    python scripts/eval_spatial_lookaround.py generate
    python scripts/eval_spatial_lookaround.py report
"""

import argparse
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

from dataloaders.avs_s4 import frame_path, wav_path  # noqa: E402
from models.megascenes_warp import megascenes_intrinsics, orbit_extrinsic  # noqa: E402
from models.novel_view_pipeline import warp_view  # noqa: E402
from models.spatial_audio import FOA_SR, direction_vector, mic_to_camera, project_point, synthesize_foa  # noqa: E402
from models.spatial_pipeline import SpatialAudioInpainter  # noqa: E402
from scripts.eval_novel_view import AUDIOS, SCENES  # noqa: E402
from scripts.eval_spatial import OBJECT_NOUNS  # noqa: E402

KAGGLE = ROOT / "data/scene-classification-images-and-audio/images"
PHOTOS = [("restaurant", KAGGLE / "restaurant3/restaurant3-43.png"), ("store", KAGGLE / "store3/store3-160.png"),
          ("village", KAGGLE / "village1/village1-154.png"), ("forest", KAGGLE / "forest3/forest662.png"),
          ("classroom", KAGGLE / "classroom5/classroom5-512.png"), ("sofa", ROOT / "examples/ecohubsofa.jpeg")]
AZ, DIST_M = 65.0, 2.5
RES = 512


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["generate", "report"])
    p.add_argument("--out", default=str(ROOT / "outputs/spatial_lookaround"))
    p.add_argument("--ckpt", default=str(ROOT / "runs/s4_v1/last.pt"))
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--location-ckpt", help="also run the trained location model (runs/loc_v2b_multiscale/last.pt)")
    return p.parse_args()


def scenes():
    out = [(sid, frame_path({"split": "test", "category": cat, "name": name}, 1)) for sid, cat, name in SCENES]
    return out + PHOTOS


def items():
    """(scene, image path, sound, side): the sound alternates over scenes so each appears on both sides."""
    sounds = list(AUDIOS)
    rows = []
    for k, (sid, path) in enumerate(scenes()):
        for j, side in enumerate(("left", "right")):
            rows.append((sid, path, sounds[(k + j) % len(sounds)], side))
            rows.append((sid, path, sounds[(k + j + 2) % len(sounds)], side))
    return rows


def generate(args):
    out = Path(args.out)
    model = SpatialAudioInpainter(args.ckpt)
    if args.location_ckpt:
        model.load_location(args.location_ckpt)
    K = megascenes_intrinsics(RES, RES)
    for i, (sid, path, sound, side) in enumerate(items()):
        tag = f"{sid}_{sound}_{side}"
        if (out / "spatial" / f"{tag}.png").is_file() and (out / "mono" / f"{tag}.png").is_file() and \
                (not args.location_ckpt or (out / "trained" / f"{tag}.png").is_file()):
            continue
        for d in ("spatial", "mono", "meta", "trained"):
            (out / d).mkdir(parents=True, exist_ok=True)
        az = AZ if side == "left" else -AZ
        mono_audio, _ = librosa.load(str(wav_path({"split": "test", "category": AUDIOS[sound][0],
                                                   "name": AUDIOS[sound][1]})), sr=FOA_SR, mono=True)
        foa = synthesize_foa(mono_audio, az, 0.0, DIST_M, seed=i)
        prompt, cls, _, _ = model.listen(foa)
        image = Image.open(path)
        plan = model.plan(image, foa, source_class=cls)
        if not (out / "spatial" / f"{tag}.png").is_file():
            spatial = model.generate(plan, prompt, seeds=(i,), steps=args.steps)[0]
            spatial.save(out / "spatial" / f"{tag}.png")
            plan["hole"].save(out / "spatial" / f"{tag}_hole.png")
        if args.location_ckpt:
            model.generate_trained(plan, prompt, seeds=(i,), steps=args.steps)[0].save(out / "trained" / f"{tag}.png")
            plan["hole"].save(out / "trained" / f"{tag}_hole.png")
            if (out / "mono" / f"{tag}.png").is_file():
                continue

        E_mono = orbit_extrinsic(20.0)
        view, hole, mask = warp_view(plan["src"], plan["depth"], E_mono)
        prompt_w, _, _ = model.semantics(foa[0], FOA_SR)
        mono = model.inpaint(view, hole, mask, prompt_w, seeds=(i,), steps=args.steps)[0]
        mono.save(out / "mono" / f"{tag}.png")
        hole.save(out / "mono" / f"{tag}_hole.png")

        p = DIST_M / 2.0 * mic_to_camera(direction_vector(az, 0.0))  # true source, scene_scale_m = 2
        truth = {"spatial": project_point(K, plan["extrinsic"], p)[:2], "mono": project_point(K, E_mono, p)[:2]}
        truth["trained"] = truth["spatial"]
        (out / "meta" / f"{tag}.json").write_text(json.dumps(
            {"scene": sid, "sound": sound, "side": side, "az": az, "heard_as": cls, "yaw_spatial": plan["yaw_deg"],
             "est_az": plan["az_deg"], "truth_uv": truth, "class": AUDIOS[sound][0]}, default=float))
        print(f"[{i + 1}/{len(items())}] {tag}: heard as {cls}, turned {plan['yaw_deg']:+.0f} deg", flush=True)


def report(args):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    out = Path(args.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    res = {m: [] for m in ("mono", "spatial", "trained") if (out / m).is_dir() and any((out / m).glob("*.png"))}
    for meta_path in sorted((out / "meta").glob("*.json")):
        m = json.loads(meta_path.read_text())
        tag = meta_path.stem
        for method in res:
            img = Image.open(out / method / f"{tag}.png").convert("RGB")
            hole = np.asarray(Image.open(out / method / f"{tag}_hole.png")) > 127
            inp = proc(text=[[f"a photo of {OBJECT_NOUNS[m['class']]}"]], images=img, return_tensors="pt").to(dev)
            with torch.no_grad():
                r = proc.post_process_grounded_object_detection(outputs=owl(**inp),
                                                                target_sizes=torch.tensor([img.size[::-1]]),
                                                                threshold=0.1)[0]
            best = None
            for b, s in zip(r["boxes"].tolist(), r["scores"].tolist()):
                x0, y0, x1, y1 = (int(np.clip(v, 0, RES - 1)) for v in b)
                if hole[y0:y1 + 1, x0:x1 + 1].mean() > 0.3 and (best is None or s > best[1]):  # mostly generated
                    best = (b, s)
            u, v = m["truth_uv"].get(method, m["truth_uv"]["spatial"])  # trained shares the spatial view
            hit = best is not None and math.isfinite(u) and best[0][0] - 16 <= u <= best[0][2] + 16
            res[method].append({"tag": tag, "side": m["side"], "found": best is not None, "hit": bool(hit)})
    summary = {}
    for method, rs in res.items():
        summary[method] = {"n": len(rs), "found": float(np.mean([r["found"] for r in rs])),
                           "hit": float(np.mean([r["hit"] for r in rs])),
                           "hit_left": float(np.mean([r["hit"] for r in rs if r["side"] == "left"])),
                           "hit_right": float(np.mean([r["hit"] for r in rs if r["side"] == "right"]))}
    (out / "summary.json").write_text(json.dumps({"summary": summary, "rows": res}, indent=2))
    for method, s in summary.items():
        print(f"{method:8s} n={s['n']}  found {100 * s['found']:.1f}%  at the sound's direction {100 * s['hit']:.1f}% "
              f"(left {100 * s['hit_left']:.1f}%, right {100 * s['hit_right']:.1f}%)")


def main():
    args = parse_args()
    generate(args) if args.command == "generate" else report(args)


if __name__ == "__main__":
    main()
