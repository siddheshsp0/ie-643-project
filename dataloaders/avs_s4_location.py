"""AVSBench S4 for learning where to draw: a hole larger than the sounding object + a heatmap of where it is.

Each item (frame 1, whose sounding-object mask is annotated in every split):
    hole    1 = fill. Training: 50% a box 1.6-3x the object's bounding box, placed at random around it;
            25% the whole image; 25% the object mask alone. Evaluation (end_holes=True): a box 3x the object's
            width and 1.6x its height with the object at one end (`end_hole_box`); otherwise a centred 2.2x box.
    heat    [1, 64, 64] Gaussian at the object's bounding-box centre with std = half its size, as spatial
            audio would report it: in training the centre is jittered by doa_sigma_px (about 4 deg of
            direction error, the error measured for models/spatial_audio.py at a 60 deg field of view) and the
            size by +-25%; 10% of items get no heatmap (all zeros).
    heat_shift  the heatmap mirrored to the other side of the hole box (validation's wrong-location control;
            only meaningful with end_holes, where the mirror lands at the empty end)
    obj     the dilated object mask (extra loss weight)
"""

import random

import numpy as np
import torch
from PIL import Image

from dataloaders.avs_s4 import AVSS4Dataset, class_balanced_subset, mask_path
from models.location_unet import gaussian_heatmap


def end_hole_box(bbox, k, res=512):
    """3x wide, 1.6x tall box with the object at its left end (k even) or right end (k odd), inside the frame.
    A model that ignores location tends to draw the object in the empty part."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    W, H = min(res, 3 * w), min(res, max(h + 16, int(1.6 * h)))
    hy0 = int(np.clip((y0 + y1) / 2 - H / 2, 0, res - H))
    for left in ((k % 2 == 0), (k % 2 == 1)):
        hx0 = x0 if left else x1 - W
        if 0 <= hx0 <= res - W:
            break
    else:
        hx0 = int(np.clip(hx0, 0, res - W))
    return int(hx0), hy0, int(hx0 + W), hy0 + H


def object_width(rec, res=512):
    m = np.asarray(Image.open(mask_path(rec, 1)).convert("L").resize((res, res), Image.NEAREST)) > 127
    if not m.any():
        return None
    xs = np.nonzero(m)[1]
    return (xs.max() - xs.min() + 1) / res


class AVSS4LocationDataset(AVSS4Dataset):
    def __init__(self, split, resolution=512, train=True, hole_probs=(0.5, 0.25, 0.25), heat_drop=0.1,
                 doa_sigma_px=31.0, size_jitter=0.25, end_holes=False, max_object_width=None, **kw):
        max_samples = kw.pop("max_samples", None)  # applied after the width filter
        super().__init__(split, resolution, train=train, **kw)
        rows = {r["name"]: f for r, f in zip(self.records, self.feat_rows)}
        if max_object_width is not None:  # keep clips whose object leaves room to choose a location
            self.records = [r for r in self.records if (object_width(r, resolution) or 1.0) <= max_object_width]
        if max_samples is not None:
            self.records = class_balanced_subset(self.records, max_samples)
        self.feat_rows = [rows[r["name"]] for r in self.records]
        self.end_holes = end_holes
        self.hole_probs = hole_probs
        self.heat_drop = heat_drop
        self.doa_sigma_px = doa_sigma_px
        self.size_jitter = size_jitter

    def hole_box(self, bbox, rng, factor=None):
        """A box `factor` times the object's bounding box (per axis) that still contains the object."""
        R = self.resolution
        x0, y0, x1, y1 = bbox
        boxes = []
        for lo, hi in ((x0, x1), (y0, y1)):
            size = hi - lo
            f = factor if factor is not None else rng.uniform(1.6, 3.0)
            new = min(R, max(size + 16, int(round(size * f))))
            start_lo, start_hi = max(0, hi - new), min(lo, R - new)
            start = (start_lo + start_hi) // 2 if factor is not None else rng.randint(start_lo, max(start_lo, start_hi))
            boxes.append((start, start + new))
        (bx0, bx1), (by0, by1) = boxes
        return bx0, by0, bx1, by1

    def __getitem__(self, i):
        rec = self.records[i]
        rng = random if self.train else random.Random(i)
        R = self.resolution
        image = self._load_image(rec, 1)
        obj = self._load_object_mask(rec, 1)
        ys, xs = np.nonzero(obj[0].numpy())
        if len(xs) == 0:  # no annotated object: plain full-image fill without a location
            bbox, kind = (0, 0, R, R), "full"
        else:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            kind = rng.choices(["box", "full", "object"], weights=self.hole_probs)[0] if self.train else "box"

        if not self.train and self.end_holes:
            box = end_hole_box(bbox, i, R)
        else:
            box = self.hole_box(bbox, rng, None if self.train else 2.2)
        hole = torch.zeros(1, R, R)
        if kind == "full":
            hole[:] = 1
        elif kind == "box":
            bx0, by0, bx1, by1 = box
            hole[:, by0:by1, bx0:bx1] = 1
        hole = torch.maximum(hole, obj)

        x0, y0, x1, y1 = bbox
        cx, cy, sx, sy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        if self.train:
            cx += rng.gauss(0, self.doa_sigma_px)
            cy += rng.gauss(0, self.doa_sigma_px)
            k = rng.uniform(1 - self.size_jitter, 1 + self.size_jitter)
            sx, sy = sx * k, sy * k
        lat = R // 8
        heat = gaussian_heatmap(cx, cy, sx, sy, lat, scale=8.0)[None]
        mirror = box[0] + box[2] - cx
        heat_shift = gaussian_heatmap(mirror, cy, sx, sy, lat, scale=8.0)[None]
        if len(xs) == 0 or (self.train and rng.random() < self.heat_drop):
            heat = torch.zeros_like(heat)

        if self.train and rng.random() < 0.5:
            image, hole, obj, heat, heat_shift = (t.flip(-1) for t in (image, hole, obj, heat, heat_shift))

        row = self.feat_rows[i]
        return {
            "intermediates": [t[row].float() for t in self.intermediates],
            "global_embed": self.global_embed[row].float(),
            "image": image, "mask": hole, "obj": obj, "heat": heat, "heat_shift": heat_shift,
            "class_idx": self.classes.index(rec["category"]), "name": rec["name"], "hole_kind": kind,
        }
