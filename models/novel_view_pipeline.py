"""Look-around pipeline: depth, warp to a new camera pose, then fill the unseen pixels with audio inpainting.

NovelViewAudioInpainter loads all the frozen models once, so the scripts can run many
image/pose/audio/seed combinations without reloading.
"""

import matplotlib
import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from PIL import Image, ImageChops, ImageFilter

from models.audio_adapter import AudioAdapter
from models.audio_token_conditioner import DEFAULT_SD_PATH, AudioTokenConditioner
from models.clap_feature_extractor import CLAPFeatureExtractor
from models.depth_anything_v2 import DEFAULT_CKPT as DEFAULT_DA_CKPT
from models.depth_anything_v2 import DepthAnythingV2Estimator, disparity_to_depth
from models.megascenes_warp import MEGASCENES_FOCAL_256, megascenes_intrinsics, warp_to_view


def center_crop_square(image):
    w, h = image.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return image.crop((left, top, left + s, top + s))


def colorize_disparity(disp):
    d = (disp - disp.min()) / max(float(disp.max() - disp.min()), 1e-8)
    rgb = matplotlib.colormaps["Spectral_r"](d)[..., :3]
    return Image.fromarray((rgb * 255).astype(np.uint8))


def warp_view(src, depth, extrinsic, focal_256=MEGASCENES_FOCAL_256, prune="stretch", max_stretch=4.0,
              prune_angle=70.0, dilate_px=8):
    """src image + its depth -> (warped RGB, hole mask, inpainting mask); masks are 'L', 255 = repaint.

    The inpainting mask is the hole mask dilated by dilate_px, because the renderer anti-aliases hole
    edges against black.
    """
    w, h = src.size
    K = megascenes_intrinsics(h, w, focal_256)
    warped, _, valid = warp_to_view(np.asarray(src), depth, K, extrinsic, prune, max_stretch, prune_angle)
    hole = Image.fromarray((~valid).astype(np.uint8) * 255)
    mask = hole.filter(ImageFilter.MaxFilter(2 * dilate_px + 1)) if dilate_px > 0 else hole
    return Image.fromarray(warped), hole, mask


class NovelViewAudioInpainter:
    def __init__(self, adapter_ckpt, sd_path=DEFAULT_SD_PATH, da_ckpt=DEFAULT_DA_CKPT, da_encoder="vitl", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.estimator = DepthAnythingV2Estimator(da_ckpt, da_encoder, device=self.device)

        ckpt = torch.load(adapter_ckpt, map_location=self.device)
        print(f"checkpoint {adapter_ckpt} (step {ckpt['step']})  layers {ckpt['layer_names']}")
        self.extractor = CLAPFeatureExtractor(layers=ckpt["layer_names"], device=self.device)
        self.adapter = AudioAdapter(**ckpt["adapter_config"]).to(self.device).eval()
        self.adapter.load_state_dict(ckpt["adapter"])
        self.conditioner = AudioTokenConditioner(sd_path, template=ckpt["template"],
                                                 num_tokens=ckpt["adapter_config"]["num_tokens"], device=self.device)
        with torch.no_grad():
            tok = self.conditioner.tokenizer
            empty = tok("", padding="max_length", max_length=tok.model_max_length, return_tensors="pt").input_ids
            self.uncond = self.conditioner.text_encoder(input_ids=empty.to(self.device)).last_hidden_state

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            sd_path, dtype=torch.float16, variant="fp16", safety_checker=None,
            feature_extractor=None, requires_safety_checker=False).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)

    def depth(self, image, res):
        """Square PIL image -> (disparity, normalized depth), both [res, res]."""
        disparity = self.estimator.disparity(image, size=(res, res))
        return disparity, disparity_to_depth(disparity)

    @torch.no_grad()
    def audio_embeds(self, wav):
        """WAV path -> prompt embeddings [1, 77, 768]."""
        feats = self.extractor.extract([str(wav)])
        return self.conditioner(self.adapter(feats["intermediates"], feats["global_embed"]))

    def inpaint(self, warped, hole, mask, prompt_embeds=None, seeds=(0,), steps=50, guidance_scale=7.5,
                feather_px=4.0):
        """Fill mask in warped, one image per seed; prompt_embeds=None uses the empty prompt.

        Seen pixels are pasted back and the generated part is blended in with a feathered edge.
        """
        cond = self.uncond if prompt_embeds is None else prompt_embeds
        n = len(seeds)
        gens = [torch.Generator(self.device).manual_seed(s) for s in seeds]
        w, h = warped.size
        images = self.pipe(prompt_embeds=cond.half().expand(n, -1, -1),
                           negative_prompt_embeds=self.uncond.half().expand(n, -1, -1),
                           image=warped, mask_image=mask, height=h, width=w, num_inference_steps=steps,
                           guidance_scale=guidance_scale, generator=gens).images
        alpha = ImageChops.lighter(mask.filter(ImageFilter.GaussianBlur(max(feather_px, 1))), hole)
        return [Image.composite(im, warped, alpha) for im in images]
