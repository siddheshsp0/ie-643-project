"""Download the frozen pre-trained weights used by this project (about 5 GB in total).

    models/stable-diffusion-inpainting/   SD-1.5 inpainting, fp16 (Approach 1 and look-around)
    models/clip-vit-large-patch14/        CLIP ViT-L/14, used only to score the evaluations
    models/depth-anything-v2/             Depth Anything V2 ViT-L (look-around; CC-BY-NC-4.0 weights)
    <laion_clap package>/630k-audioset-best.pt   LAION-CLAP audio encoder (read from the package folder)

The trained audio adapter is not downloaded: it is in the repository at runs/s4_v1/last.pt.

Usage (from the project root):
    python scripts/download_weights.py                         # everything
    python scripts/download_weights.py --skip-clip --skip-depth   # only what Approach-1 inference needs
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
SD_REPO = "stable-diffusion-v1-5/stable-diffusion-inpainting"
SD_FILES = [
    "model_index.json", "scheduler/scheduler_config.json",
    "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
    "tokenizer/merges.txt", "tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
]
CLIP_REPO = "openai/clip-vit-large-patch14"
CLIP_FILES = ["config.json", "merges.txt", "model.safetensors", "preprocessor_config.json",
              "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]
DA_REPO, DA_FILE = "depth-anything/Depth-Anything-V2-Large", "depth_anything_v2_vitl.pth"
CLAP_REPO, CLAP_FILE = "lukewys/laion_clap", "630k-audioset-best.pt"


def fetch(repo, files, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        if (dest / f).is_file():
            print(f"have  {dest / f}")
            continue
        hf_hub_download(repo, f, local_dir=dest)
        print(f"got   {dest / f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-sd", action="store_true")
    p.add_argument("--skip-clip", action="store_true", help="CLIP is only needed by the evaluation scripts")
    p.add_argument("--skip-depth", action="store_true", help="Depth Anything is only needed by the look-around code")
    p.add_argument("--skip-clap", action="store_true")
    args = p.parse_args()

    if not args.skip_sd:
        fetch(SD_REPO, SD_FILES, ROOT / "models/stable-diffusion-inpainting")
    if not args.skip_clip:
        fetch(CLIP_REPO, CLIP_FILES, ROOT / "models/clip-vit-large-patch14")
    if not args.skip_depth:
        fetch(DA_REPO, [DA_FILE], ROOT / "models/depth-anything-v2")
    if not args.skip_clap:
        try:
            import laion_clap
        except ImportError:
            sys.exit("laion_clap is not installed; run pip install -r requirements.txt first")
        fetch(CLAP_REPO, [CLAP_FILE], Path(laion_clap.__file__).parent)


if __name__ == "__main__":
    main()
