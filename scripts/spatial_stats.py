"""Paired comparisons for the spatial-audio evaluations, with bootstrap 95% intervals.

Reads the per-case results written by `scripts/eval_spatial.py` (metrics.csv) and
`scripts/eval_spatial_lookaround.py` (summary.json rows) and prints, for each pair of methods, the difference in
hit rate on the same cases with a 95% interval from resampling the cases (paired bootstrap).

    python scripts/spatial_stats.py
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def paired_diff(a, b, n=10000, seed=0):
    """Mean of a - b over shared cases, with a 95% paired-bootstrap interval (in percentage points)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return 100 * d.mean(), 100 * lo, 100 * hi


def load_in_place(paths):
    """{condition: {clip: row}} from one or more metrics.csv files."""
    out = {}
    for p in paths:
        for row in csv.DictReader(open(p)):
            out.setdefault(row["condition"], {})[row["name"]] = row
    return out


def in_place(args):
    paths = [ROOT / args.in_place / "metrics.csv"]
    trained = {}
    for tag, d in [("trained, input channel", args.trained_a), ("trained, multiscale", args.trained_b)]:
        rows = load_in_place([ROOT / d / "metrics.csv"])
        trained[tag] = rows["trained"]
    res = load_in_place(paths)
    res.update(trained)
    names = sorted(set.intersection(*(set(v) for v in res.values())))
    print(f"In place: {len(names)} S4 test clips")
    for metric in ["hit", "iou_hit", "clip_top1"]:
        print(f"  {metric}")
        for m in res:
            print(f"    {m:24s} {100 * np.mean([int(res[m][c][metric]) for c in names]):5.1f}%")
        for a, b in [("steer", "mono"), ("free", "mono"), ("steer", "free"), ("trained, input channel", "mono"),
                     ("trained, multiscale", "mono"), ("trained, multiscale", "steer")]:
            d, lo, hi = paired_diff([int(res[a][c][metric]) for c in names], [int(res[b][c][metric]) for c in names])
            print(f"    {a} - {b}: {d:+.1f} pts [{lo:+.1f}, {hi:+.1f}]")


def lookaround(args):
    rows = json.load(open(ROOT / args.lookaround / "summary.json"))["rows"]
    hits = {m: {r["tag"]: int(r["hit"]) for r in rs} for m, rs in rows.items()}
    tags = sorted(set.intersection(*(set(v) for v in hits.values())))
    print(f"Around the corner: {len(tags)} cases")
    for m in hits:
        print(f"    {m:24s} {100 * np.mean([hits[m][t] for t in tags]):5.1f}%")
    for a, b in [("spatial", "mono"), ("trained", "mono"), ("spatial", "trained")]:
        if a in hits and b in hits:
            d, lo, hi = paired_diff([hits[a][t] for t in tags], [hits[b][t] for t in tags])
            print(f"    {a} - {b}: {d:+.1f} pts [{lo:+.1f}, {hi:+.1f}]")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in-place", default="outputs/spatial_eval")
    p.add_argument("--trained-a", default="outputs/spatial_eval_trained_v2a")
    p.add_argument("--trained-b", default="outputs/spatial_eval_trained_v2b")
    p.add_argument("--lookaround", default="outputs/spatial_lookaround")
    args = p.parse_args()
    in_place(args)
    lookaround(args)


if __name__ == "__main__":
    main()
