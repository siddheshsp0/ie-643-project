"""Frozen Depth Anything V2 (ViT-L) depth, processed the same way MegaScenes processes Depth Anything v1.

The model predicts relative disparity. Following MegaScenes' video_script.py, we invert it and divide by
the 20th-percentile depth, so near content sits around depth 1, the unit the camera poses use.
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DA2_REPO = ROOT / "third_party/Depth-Anything-V2"
DEFAULT_CKPT = ROOT / "models/depth-anything-v2/depth_anything_v2_vitl.pth"
MODEL_CONFIGS = {  # from Depth-Anything-V2/run.py
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def disparity_to_depth(disparity, disparity_min=1e-3, quantile=0.2, max_depth=100.0):
    """[H, W] disparity -> [H, W] depth, normalized as in MegaScenes (20th percentile = 1, clipped at 100)."""
    depth = 1.0 / np.maximum(np.asarray(disparity, dtype=np.float64), disparity_min)
    depth = depth / np.quantile(depth.reshape(-1), q=quantile)
    return depth.clip(0.0, max_depth).astype(np.float32)


class DepthAnythingV2Estimator:
    def __init__(self, ckpt_path=DEFAULT_CKPT, encoder="vitl", input_size=518, device=None):
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"Depth Anything V2 checkpoint not found: {ckpt_path}")
        if str(DA2_REPO) not in sys.path:
            sys.path.insert(0, str(DA2_REPO))
        logging.getLogger("dinov2").setLevel(logging.ERROR)  # "xFormers not available" is expected
        from depth_anything_v2.dpt import DepthAnythingV2

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
        self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def disparity(self, image, size=None):
        """PIL RGB image -> relative disparity [H, W], larger = closer, resized bicubically to size."""
        rgb = np.asarray(image.convert("RGB"))
        h, w = size if size is not None else rgb.shape[:2]
        x, _ = self.model.image2tensor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), self.input_size)
        disp = self.model(x.to(self.device)).float()  # [1, h', w']
        disp = F.interpolate(disp[None], (h, w), mode="bicubic", align_corners=False)[0, 0]
        return disp.cpu().numpy()

    def depth(self, image, size=None):
        """PIL RGB image -> normalized depth [H, W]."""
        return disparity_to_depth(self.disparity(image, size))
