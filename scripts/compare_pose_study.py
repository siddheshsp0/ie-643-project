"""Compare eval_novel_view.py runs that differ only in the camera pose and write POSE_STUDY.md.

For each run and yaw it reports hole geometry and CLIP hit rates, with 95% intervals bootstrapped over
scenes. "Roomy" is the part of the hole more than 32 px from any seen pixel, i.e. space for a new object.

Usage (from the project root, after `eval_novel_view.py report` for each run):
    python scripts/compare_pose_study.py v1 v2_noforward v2_rotonly v2_focal200 v2_yaw20
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs/novel_view_eval"
TARGETS = {"cat": "cat_meowing", "dog": "dog_barking", "baby": "baby_laughter", "guitar": "playing_acoustic_guitar"}
ROOMY_PX = 32


def hole_stats(hole):
    pad = np.pad(hole, 64, constant_values=True)  # outside the frame counts as hole
    dist = ndimage.distance_transform_edt(pad)[64:-64, 64:-64]
    lab, _ = ndimage.label(hole)
    border = np.setdiff1d(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]), [0])
    frame = np.isin(lab, border)
    return {"hole": hole.mean(), "frame": frame.mean(), "interior": (hole & ~frame).mean(),
            "roomy": (dist >= ROOMY_PX).mean()}


def bootstrap_ci(per_scene, n=2000, seed=0):
    """95% CI of the pooled hit rate, resampling whole scenes."""
    rng = np.random.default_rng(seed)
    means = [np.concatenate([per_scene[i] for i in rng.integers(0, len(per_scene), len(per_scene))]).mean()
             for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def paired_delta(cell, ref, n=2000, seed=0):
    """Per-scene paired difference in hit rate (cell - ref), its mean and 95% bootstrap CI."""
    d = np.array([cell["per_scene"][s] - ref["per_scene"][s] for s in cell["per_scene"] if s in ref["per_scene"]])
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(boots, [2.5, 97.5])


def reference_cell(ref_run, yaw):
    """Reference cell at the same yaw, or else the same direction (e.g. +20 is compared with +15)."""
    if yaw in ref_run:
        return ref_run[yaw]
    same_side = [y for y in ref_run if np.sign(float(y)) == np.sign(float(yaw))]
    return ref_run[same_side[0]] if same_side else None


def describe(run):
    gen = json.loads((EVAL / run / "generate_args.json").read_text())
    pose, focal = gen.get("pose", "orbit"), gen.get("focal_256", 80.0)
    return {"orbit": "orbit", "no-forward": "orbit, no forward move", "rotation-only": "rotation only"}[pose] + \
        ("" if focal == 80.0 else f", focal {focal:g}")


def read_rows(run):
    return list(csv.DictReader(open(EVAL / run / "metrics.csv")))


def analyse(run, seeds):
    rows = [r for r in read_rows(run) if r["seed"] in seeds]
    none = {(r["scene"], r["yaw"], r["seed"]): r for r in rows if r["condition"] == "none"}
    out = {}
    for yaw in sorted({r["yaw"] for r in rows}, key=lambda y: -float(y)):
        audio = [r for r in rows if r["yaw"] == yaw and r["condition"] in TARGETS]
        scenes = sorted({r["scene"] for r in audio})
        hits = {s: np.array([r["top1"] == TARGETS[r["condition"]] for r in audio if r["scene"] == s], float)
                for s in scenes}
        base = [none[(r["scene"], yaw, r["seed"])] for r in audio]
        geo = [hole_stats(np.asarray(Image.open(EVAL / run / f"scenes/{s}/yaw{yaw}/mask.png")) > 127) for s in scenes]
        out[yaw] = {
            "label": describe(run),
            **{k: float(np.mean([g[k] for g in geo])) for k in geo[0]},
            "top1_audio": float(np.concatenate(list(hits.values())).mean()),
            "ci": bootstrap_ci(list(hits.values())),
            "top1_no_audio": float(np.mean([b["top1"] == TARGETS[r["condition"]] for r, b in zip(audio, base)])),
            "win_rate": float(np.mean([float(r[f"p_{r['condition']}"]) > float(b[f"p_{r['condition']}"])
                                       for r, b in zip(audio, base)])),
            "per_sound": {c: float(np.mean([r["top1"] == TARGETS[c] for r in audio if r["condition"] == c]))
                          for c in TARGETS},
            "per_scene": {s: float(h.mean()) for s, h in hits.items()},
            "scene_roomy": {s: g["roomy"] for s, g in zip(scenes, geo)},
        }
    return out


def pct(x):
    return f"{100 * x:.0f}%"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories under outputs/novel_view_eval (first = reference)")
    p.add_argument("--out", default=str(EVAL / "POSE_STUDY.md"))
    args = p.parse_args()

    # only use seeds that every run has (some runs were made with 2 seeds, some with 3)
    seeds = sorted(set.intersection(*({r["seed"] for r in read_rows(run)} for run in args.runs)), key=int)
    results = {run: analyse(run, set(seeds)) for run in args.runs}
    cells = [(run, yaw, v) for run, by_yaw in results.items() for yaw, v in by_yaw.items()]
    scenes = sorted({s for _, _, v in cells for s in v["per_scene"]})

    L = ["# Pose study: which camera move gives the sound room to act?", "",
         f"Generated by `scripts/compare_pose_study.py {' '.join(args.runs)}`. Same 8 S4 test scenes, 4 sounds "
         f"(cat, dog, baby, guitar) and seeds {', '.join(seeds)} in every run, so the runs differ only in the warp. "
         f"Each cell has {8 * 4 * len(seeds)} audio images (8 scenes × 4 sounds × {len(seeds)} seeds; runs with "
         "more seeds are restricted to the shared ones). Hit = CLIP top-1 on the generated pixels is the sound's "
         "class. The 95% interval bootstraps over scenes.", "",
         "Hole columns are fractions of the 512×512 view. **frame** = holes touching the border (the band that "
         "rotates into view). **interior** = disocclusions behind near objects. **roomy** = hole pixels more than "
         f"{ROOMY_PX} px from any seen pixel (room for a 64 px object).", "",
         f"**Δ vs {args.runs[0]}** = paired difference in audio top-1 over the same scenes, against "
         f"{args.runs[0]} at the same yaw (or the same direction, e.g. +20° vs +15°); points, 95% CI over scenes.", "",
         "| run | pose | yaw | unseen | frame | interior | roomy | **top-1 audio** (95% CI) | "
         f"Δ vs {args.runs[0]} | top-1 no audio | audio > no audio (paired) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, yaw, v in cells:
        ref = reference_cell(results[args.runs[0]], yaw)
        if run == args.runs[0] or ref is None:
            delta = "–"
        else:
            d, (lo, hi) = paired_delta(v, ref)
            delta = f"{100 * d:+.0f} ({100 * lo:+.0f} to {100 * hi:+.0f})"
        L.append(f"| {run} | {v['label']} | {yaw}° | {pct(v['hole'])} | {pct(v['frame'])} | {pct(v['interior'])} | "
                 f"{pct(v['roomy'])} | **{pct(v['top1_audio'])}** ({pct(v['ci'][0])}–{pct(v['ci'][1])}) | {delta} | "
                 f"{pct(v['top1_no_audio'])} | {pct(v['win_rate'])} |")
    per_view = [(v["per_scene"][s], r) for _, _, v in cells for s, r in v["scene_roomy"].items()]
    hit, roomy = np.array(per_view).T
    rho = np.corrcoef(np.argsort(np.argsort(hit)), np.argsort(np.argsort(roomy)))[0, 1]
    L += ["", f"Across all {len(per_view)} scene-views, Spearman correlation between roomy hole area and audio "
          f"top-1 = **{rho:+.2f}**."]
    L += ["", "## Top-1 hit rate per sound", "", "| run | yaw | " + " | ".join(TARGETS) + " |",
          "|---|---|" + "---|" * len(TARGETS)]
    L += [f"| {run} | {yaw}° | " + " | ".join(pct(v["per_sound"][c]) for c in TARGETS) + " |" for run, yaw, v in cells]
    L += ["", "## Top-1 hit rate per scene (audio images)", "",
          "| scene | " + " | ".join(f"{run} {yaw}°" for run, yaw, _ in cells) + " |", "|---|" + "---|" * len(cells)]
    L += [f"| {s} | " + " | ".join(pct(v["per_scene"][s]) if s in v["per_scene"] else "–" for _, _, v in cells) + " |"
          for s in scenes]
    L += ["", "Contact sheets: `outputs/novel_view_eval/<run>/sheets/<scene>.png`. Per-run reports: "
          "`outputs/novel_view_eval/<run>/README.md`."]
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
