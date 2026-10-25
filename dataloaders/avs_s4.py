"""AVSBench single-source (S4) data, paired with the cached CLAP features, for training the adapter.

Files per split (listed in s4_meta_data.csv):
    s4_data/audio_wav/{split}/{category}/{name}.wav
    s4_data/visual_frames/{split}/{category}/{name}/{name}_{1..5}.png   224x224, one frame per second
    s4_data/gt_masks/{split}/{category}/{name}/{name}_{k}.png           train only has a mask for k=1
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
S4_ROOT = ROOT / "data/avsbench_data_processed/Single-source"
S4_DATA = S4_ROOT / "s4_data"
S4_CSV = S4_ROOT / "s4_meta_data.csv"
CACHE_DIR = ROOT / "data/cache/clap_s4"
NUM_FRAMES = 5
MASK_TYPES = ("object", "random", "full")


def read_split(split):
    with open(S4_CSV) as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def all_classes():
    with open(S4_CSV) as f:
        return sorted({r["category"] for r in csv.DictReader(f)})


def wav_path(rec):
    return S4_DATA / "audio_wav" / rec["split"] / rec["category"] / f"{rec['name']}.wav"


def frame_path(rec, k):
    return S4_DATA / "visual_frames" / rec["split"] / rec["category"] / rec["name"] / f"{rec['name']}_{k}.png"


def mask_path(rec, k):
    return S4_DATA / "gt_masks" / rec["split"] / rec["category"] / rec["name"] / f"{rec['name']}_{k}.png"


def label_text(category):
    return category.replace("_", " ")


def class_balanced_subset(records, n):
    """First n records taken round-robin over sorted categories (the CSV is grouped by class)."""
    by_class = {}
    for r in records:
        by_class.setdefault(r["category"], []).append(r)
    queues = [by_class[c] for c in sorted(by_class)]
    out, depth = [], 0
    while len(out) < n and any(depth < len(q) for q in queues):
        out += [q[depth] for q in queues if depth < len(q)]
        depth += 1
    return out[:n]


class AVSS4Dataset(Dataset):
    """S4 clips paired with cached CLAP features.

    train=True samples a mask type per item with `mask_probs` (object / random box / full)
    and applies horizontal flips. train=False is deterministic: frame 1, its object mask, no flip.
    """

    def __init__(self, split, resolution=512, train=True, mask_probs=(0.6, 0.2, 0.2),
                 dilate_px=8, max_samples=None, cache_dir=CACHE_DIR):
        cache_file = Path(cache_dir) / f"{split}.pt"
        if not cache_file.is_file():
            raise FileNotFoundError(f"{cache_file} missing; run scripts/cache_clap_features.py first")
        cache = torch.load(cache_file, map_location="cpu")
        index = {n: i for i, n in enumerate(cache["names"])}

        records = read_split(split)
        missing = [r["name"] for r in records if r["name"] not in index]
        if missing:
            raise RuntimeError(f"{len(missing)} {split} clips have no cached features, e.g. {missing[:3]}")
        if max_samples is not None:
            records = class_balanced_subset(records, max_samples)

        self.split = split
        self.records = records
        self.feat_rows = [index[r["name"]] for r in records]
        self.intermediates = cache["intermediates"]  # list of [N, T_i, D_i] fp16
        self.global_embed = cache["global_embed"]  # [N, 512] fp16
        self.layer_names = cache["layer_names"]
        self.classes = all_classes()
        self.resolution = resolution
        self.train = train
        self.mask_probs = mask_probs
        self.dilate_px = dilate_px
        # Only frame 1 is annotated in train; every frame is annotated in val/test.
        self.mask_frames = (1,) if split == "train" else tuple(range(1, NUM_FRAMES + 1))

    def __len__(self):
        return len(self.records)

    def _load_image(self, rec, k):
        img = Image.open(frame_path(rec, k)).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        return x.clamp(-1.0, 1.0)

    def _load_object_mask(self, rec, k):
        m = Image.open(mask_path(rec, k)).convert("L").resize((self.resolution, self.resolution), Image.NEAREST)
        m = torch.from_numpy((np.asarray(m) > 127).astype(np.float32))[None]
        if self.dilate_px > 0:
            k_size = 2 * self.dilate_px + 1
            m = F.max_pool2d(m[None], k_size, stride=1, padding=self.dilate_px)[0]
        return m

    def _random_box_mask(self, rng):
        R = self.resolution
        area = rng.uniform(0.1, 0.5)
        aspect = rng.uniform(0.5, 2.0)
        h = min(R, int(round(R * (area * aspect) ** 0.5)))
        w = min(R, int(round(R * (area / aspect) ** 0.5)))
        top, left = rng.randint(0, R - h), rng.randint(0, R - w)
        m = torch.zeros(1, R, R)
        m[:, top:top + h, left:left + w] = 1.0
        return m

    def __getitem__(self, i):
        rec = self.records[i]
        rng = random if self.train else random.Random(i)
        if self.train:
            mask_type = rng.choices(MASK_TYPES, weights=self.mask_probs)[0]
            frame = rng.choice(self.mask_frames) if mask_type == "object" else rng.randint(1, NUM_FRAMES)
        else:
            mask_type, frame = "object", 1

        image = self._load_image(rec, frame)
        if mask_type == "object":
            mask = self._load_object_mask(rec, frame)
        elif mask_type == "random":
            mask = self._random_box_mask(rng)
        else:
            mask = torch.ones(1, self.resolution, self.resolution)

        if self.train and rng.random() < 0.5:
            image, mask = image.flip(-1), mask.flip(-1)

        row = self.feat_rows[i]
        return {
            "intermediates": [t[row].float() for t in self.intermediates],
            "global_embed": self.global_embed[row].float(),
            "image": image,
            "mask": mask,
            "class_idx": self.classes.index(rec["category"]),
            "name": rec["name"],
            "frame": frame,
            "mask_type": mask_type,
        }
