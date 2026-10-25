"""Run the frozen CLAP encoder once over every S4 clip and save the features for training.

Writes data/cache/clap_s4/{split}.pt with the clip names, the CLAP layer names, the intermediate
features (list of [N, T_i, D_i]) and the global embeddings ([N, 512]), both in float16.

Usage (from the project root):
    python scripts/cache_clap_features.py [--splits train val test] [--batch-size 32]
"""

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloaders.avs_s4 import CACHE_DIR, read_split, wav_path  # noqa: E402
from models.clap_feature_extractor import CLAPFeatureExtractor  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-dir", default=str(CACHE_DIR))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = CLAPFeatureExtractor()
    print(f"CLAP: {extractor.ckpt_path}\nlayers: {extractor.layer_names}  dims={extractor.feature_dims}  "
          f"T={extractor.seq_lens}  global={extractor.global_dim}  device={extractor.device}")

    for split in args.splits:
        out_file = out_dir / f"{split}.pt"
        if out_file.exists() and not args.overwrite:
            print(f"[{split}] {out_file} exists, skipping (use --overwrite)")
            continue
        records = read_split(split)
        names, inter_chunks, global_chunks = [], [], []
        t0 = time.time()
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            out = extractor.extract([str(wav_path(r)) for r in batch])
            for t in out["intermediates"] + [out["global_embed"]]:
                if not torch.isfinite(t).all():
                    raise RuntimeError(f"non-finite CLAP features in batch starting at {batch[0]['name']}")
            names += [r["name"] for r in batch]
            inter_chunks.append([t.half().cpu() for t in out["intermediates"]])
            global_chunks.append(out["global_embed"].half().cpu())
            done = start + len(batch)
            if done % (args.batch_size * 20) == 0 or done == len(records):
                print(f"[{split}] {done}/{len(records)}  {time.time() - t0:.0f}s")

        intermediates = [torch.cat([c[i] for c in inter_chunks]) for i in range(len(extractor.layer_names))]
        global_embed = torch.cat(global_chunks)
        torch.save({"names": names, "layer_names": list(extractor.layer_names), "intermediates": intermediates,
                    "global_embed": global_embed, "ckpt_path": extractor.ckpt_path}, out_file)
        shapes = ", ".join(str(tuple(t.shape)) for t in intermediates)
        print(f"[{split}] saved {out_file}: intermediates {shapes}, global {tuple(global_embed.shape)}")


if __name__ == "__main__":
    main()
