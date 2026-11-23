"""Training-free spatial-audio conditioning: FOA tells the model where the source is, not only what it is.

    FOA [W, Y, Z, X] --W--> CLAP -> AudioAdapter -> CLIP prompt               what   (Approach 1, unchanged)
                     --intensity vector, diffuseness--> direction, distance     where, how far
    photo --Depth Anything V2--> depth
    direction --> camera yaw that faces the source (MegaScenes orbit) --> warp, unseen-pixel mask
    source point --project into the new view--> (u, v, depth) --class size prior--> location box
    location = box AND (unseen OR scene farther than the source)     nearer surfaces stay in front (occlusion)
    steer (default): one pass over the unseen pixels and the location with the audio prompt, its cross-attention
                     steered to a heatmap at the box (models/attention_steering.py)
    mask:            pass 1 fills the unseen pixels with no audio (background), pass 2 the location with audio;
                     on S4 test clips it drew the object far less often than steering

Distances: depth from Depth Anything is relative (20th-percentile depth = 1, as in MegaScenes); scene_scale_m
says how many metres that unit is (default 2 m, an indoor guess). Audio distances are in metres.
Nothing here is trained; the audio adapter is runs/s4_v1.
"""

import math

import librosa
import numpy as np
import torch
from diffusers import UNet2DConditionModel
from PIL import Image, ImageDraw, ImageFilter
from transformers import RobertaTokenizer

from models.attention_steering import steer_cross_attention
from models.location_unet import gaussian_heatmap, load_location_state, sample_location

from models.megascenes_warp import MEGASCENES_FOCAL_256, megascenes_intrinsics, orbit_extrinsic, warp_to_view
from models.novel_view_pipeline import NovelViewAudioInpainter, center_crop_square
from models.spatial_audio import (FOA_SR, analyse_foa, beamform, direction_vector, estimate_distance, half_fov_deg,
                                  mic_to_camera, project_point, yaw_to_face)

# Typical height in metres and width / height of what the model draws for each AVSBench S4 class
# (rough, hand-set priors; e.g. an instrument class usually comes out as a person playing it).
SIZE_PRIORS = {
    "ambulance_siren": (2.6, 2.2), "baby_laughter": (0.7, 0.8), "cap_gun_shooting": (0.3, 1.6),
    "cat_meowing": (0.35, 1.3), "chainsawing_trees": (0.5, 1.8), "coyote_howling": (0.7, 1.4),
    "dog_barking": (0.6, 1.3), "driving_buses": (3.0, 2.5), "female_singing": (1.65, 0.45),
    "helicopter": (3.5, 3.0), "horse_clip-clop": (1.7, 1.3), "lawn_mowing": (1.0, 1.2),
    "lions_roaring": (1.1, 1.6), "male_speech": (1.75, 0.45), "mynah_bird_singing": (0.25, 1.0),
    "playing_acoustic_guitar": (1.0, 0.8), "playing_glockenspiel": (0.4, 1.5), "playing_piano": (1.3, 1.2),
    "playing_tabla": (0.4, 1.2), "playing_ukulele": (0.6, 0.6), "playing_violin": (1.0, 0.7),
    "race_car": (1.2, 3.0), "typing_on_computer_keyboard": (0.4, 1.5),
}
CLASSES = sorted(SIZE_PRIORS)
NEGATIVE_PROMPT = "text, caption, watermark, logo, letters, blurry"
# the background pass must not invent a foreground object of its own next to the source
BACKGROUND_NEGATIVE = "person, people, animal, " + NEGATIVE_PROMPT


def make_box(u, v, h_px, w_px, res, min_frac=0.3, max_frac=0.8):
    """Box centred on the projected source, its physical size floored at min_frac of the image: with a wide
    lens a distant source covers a few dozen pixels, too few for Stable Diffusion to draw an object."""
    scale = max(1.0, min_frac * res / max(h_px, 1e-6), 0.66 * min_frac * res / max(w_px, 1e-6))
    h = float(np.clip(h_px * scale, 0, max_frac * res))
    w = float(np.clip(w_px * scale, 0, max_frac * res))
    cx = float(np.clip(u, w / 2, res - w / 2))
    cy = float(np.clip(v, h / 2, res - h / 2))
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class SpatialAudioInpainter(NovelViewAudioInpainter):
    def __init__(self, adapter_ckpt, **kwargs):
        super().__init__(adapter_ckpt, **kwargs)
        tok = RobertaTokenizer.from_pretrained("roberta-base")
        text = tok([f"the sound of {c.replace('_', ' ')}" for c in CLASSES], padding="max_length", truncation=True,
                   max_length=77, return_tensors="pt")
        with torch.no_grad():
            self.class_text = self.extractor.clap.get_text_embedding(dict(text))  # [23, 512], unit norm
            self.negative, self.background_negative = (self.encode_text(t) for t in (NEGATIVE_PROMPT,
                                                                                      BACKGROUND_NEGATIVE))

    @torch.no_grad()
    def encode_text(self, text):
        tok = self.conditioner.tokenizer
        ids = tok(text, padding="max_length", max_length=tok.model_max_length, truncation=True,
                  return_tensors="pt").input_ids
        return self.conditioner.text_encoder(input_ids=ids.to(self.device)).last_hidden_state

    def listen(self, foa, sr=FOA_SR):
        """FOA -> (prompt embeds, class, class prob, direction/diffuseness analysis).

        The semantics come from a hypercardioid beam steered to the estimated direction, not from W: in a
        reverberant room the beam keeps the direct sound and cuts the reverberation, which CLAP needs
        (zero-shot accuracy on S4 test clips at 2-4 m: W 70-74%, beam 76-83%, clean mono 91%).
        """
        audio = analyse_foa(foa, sr)
        return (*self.semantics(beamform(foa, audio["az_deg"], audio["el_deg"]), sr), audio)

    @torch.no_grad()
    def semantics(self, mono, sr):
        """Mono waveform -> (prompt embeds [1, 77, 768], CLAP zero-shot class, its prob)."""
        y = librosa.resample(np.asarray(mono, dtype=np.float32), orig_sr=sr, target_sr=48000) if sr != 48000 else mono
        feats = self.extractor.extract_from_waveforms([torch.from_numpy(np.asarray(y, dtype=np.float32))])
        embeds = self.conditioner(self.adapter(feats["intermediates"], feats["global_embed"]))
        probs = (100.0 * feats["global_embed"] @ self.class_text.T).softmax(-1)[0]
        k = int(probs.argmax())
        return embeds, CLASSES[k], float(probs[k])

    def plan(self, image, foa, sr=FOA_SR, res=512, yaw_deg=None, max_yaw_deg=30.0, margin_deg=25.0,
             focal_256=MEGASCENES_FOCAL_256, scene_scale_m=2.0, source_class=None, size_m=None, distance_m=None,
             dilate_px=8, prune="stretch", min_box_frac=0.4):
        """Where the source goes. image: PIL (any size, centre-cropped); foa: [4, N] ACN/SN3D."""
        image = center_crop_square(image.convert("RGB"))
        src = image.resize((res, res), Image.LANCZOS)
        disparity, depth = self.depth(image, res)
        audio = analyse_foa(foa, sr)
        r_m = distance_m if distance_m is not None else estimate_distance(audio["diffuseness"])
        K = megascenes_intrinsics(res, res, focal_256)
        if yaw_deg is None:
            yaw_deg = yaw_to_face(audio["az_deg"], K, res, margin_deg, max_yaw_deg)
        E = orbit_extrinsic(yaw_deg, tz=abs(math.sin(math.radians(yaw_deg))))  # turn and step back, never forward
        if yaw_deg:
            warped_np, depth_t, valid = warp_to_view(np.asarray(src), depth, K, E, prune)
            warped = Image.fromarray(warped_np)
        else:
            warped, depth_t, valid = src, depth, np.ones(depth.shape, bool)
        hole = ~valid

        p = (r_m / scene_scale_m) * mic_to_camera(direction_vector(audio["az_deg"], audio["el_deg"]))
        u, v, z = project_point(K, E, p)
        visible = math.isfinite(u) and 0 <= u < res and 0 <= v < res
        if not math.isfinite(u):  # behind the new camera: hug the edge on the source's side
            u, v, z = (0.0 if audio["az_deg"] > 0 else res - 1.0), res / 2, max(z, 1e-3)
        height_m, aspect = SIZE_PRIORS.get(source_class, (1.0, 1.0))
        height_m = size_m or height_m
        h_px = K[1, 1] * height_m / (max(z, 1e-3) * scene_scale_m)
        box = make_box(u, v, h_px, h_px * aspect, res, min_box_frac)

        box_mask = np.zeros((res, res), bool)
        x0, y0, x1, y1 = (int(round(c)) for c in box)
        box_mask[y0:y1, x0:x1] = True
        behind = depth_t > z  # scene surface farther than the source: the source covers it
        location = box_mask & (hole | behind)
        occluded = location.sum() < 0.15 * box_mask.sum()
        if occluded:  # the source sits behind nearer geometry; draw what can show through the unseen pixels
            location = box_mask & hole if (box_mask & hole).any() else box_mask
        loc = Image.fromarray(location.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(9))
        hole_img = Image.fromarray(hole.astype(np.uint8) * 255)
        inpaint_mask = hole_img.filter(ImageFilter.MaxFilter(2 * dilate_px + 1)) if dilate_px > 0 else hole_img
        return {"src": src, "disparity": disparity, "depth": depth, "warped": warped, "hole": hole_img, "inpaint_mask": inpaint_mask,
                "location": loc, "box": box, "yaw_deg": float(yaw_deg), "extrinsic": E, "K": K,
                "u": u, "v": v, "z": z, "visible": bool(visible), "occluded": bool(occluded),
                "az_deg": audio["az_deg"], "el_deg": audio["el_deg"], "diffuseness": audio["diffuseness"],
                "distance_m": r_m, "half_fov_deg": half_fov_deg(K, res)}

    def fill(self, image, mask, keep, prompt_embeds=None, seed=0, steps=50, guidance_scale=7.5, negative=True,
             feather_px=4.0):
        """Inpaint `mask` in `image` (one seed); pixels outside max(blurred mask, keep) are copied unchanged.
        negative: True = the text/watermark negative, False = the empty prompt, or prompt embeds."""
        cond = self.uncond if prompt_embeds is None else prompt_embeds
        neg = negative if torch.is_tensor(negative) else (self.negative if negative else self.uncond)
        w, h = image.size
        out = self.pipe(prompt_embeds=cond.half(), negative_prompt_embeds=neg.half(), image=image, mask_image=mask,
                        height=h, width=w, num_inference_steps=steps, guidance_scale=guidance_scale,
                        generator=torch.Generator(self.device).manual_seed(seed)).images[0]
        alpha = Image.fromarray(np.maximum(np.asarray(mask.filter(ImageFilter.GaussianBlur(feather_px))),
                                           np.asarray(keep)))
        return Image.composite(out, image, alpha)

    def load_location(self, ckpt_path):
        """Load a trained location model (scripts/train_location.py) for generate_trained."""
        ckpt = torch.load(ckpt_path, map_location="cpu")
        unet = UNet2DConditionModel.from_pretrained(self.sd_path, subfolder="unet", variant="fp16",
                                                    torch_dtype=torch.float32)
        load_location_state(unet, ckpt["location"], rank=ckpt["lora_rank"], alpha=float(ckpt["lora_rank"]))
        self.location_unet = unet.to(self.device, torch.float16).eval()
        self.location_step = ckpt["step"]

    @staticmethod
    def heatmap(box, res, device):
        """Location heatmap at latent size: a Gaussian centred on the box, std = half its width / height."""
        x0, y0, x1, y1 = box
        return gaussian_heatmap((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, res // 8, scale=8.0,
                                device=device)[None, None]

    @torch.no_grad()
    def generate_trained(self, plan, prompt_embeds, seeds=(0,), steps=50, guidance_scale=7.5, negative=True):
        """One pass with the trained location model: the unseen pixels (plus the source's location) are filled,
        with the heatmap at the projected source telling the U-Net where the sound is."""
        res = plan["warped"].size[0]
        to_t = lambda im: torch.from_numpy(np.asarray(im.convert("RGB"), np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
        image = to_t(plan["warped"]).to(self.device)
        mask = np.maximum(np.asarray(plan["inpaint_mask"]), np.asarray(plan["location"])) > 127
        mask = torch.from_numpy(mask.astype(np.float32))[None, None].to(self.device)
        heat = self.heatmap(plan["box"], res, self.device)
        neg = self.negative if negative else self.uncond
        images = []
        for s in seeds:
            out = sample_location(self.location_unet, self.pipe.vae, self.pipe.scheduler.config, prompt_embeds,
                                  neg, image, mask, heat, steps, guidance_scale, seed=s)
            arr = ((out[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(arr))
        return images

    def fill_steered(self, image, mask, keep, prompt_embeds, heat, seed=0, steps=50, guidance_scale=7.5,
                     negative=True, floor=0.1, boost=1.0, active_frac=0.5, feather_px=4.0):
        """One pass over `mask` with the audio prompt, its cross-attention steered to `heat` ([64, 64], peak 1)
        for the first active_frac of the steps (models/attention_steering.py)."""
        state = {"heat": heat.to(self.device), "active": True, "first_key": self.conditioner.start, "floor": floor,
                 "boost": boost, "cfg": guidance_scale > 1}

        def on_step_end(pipe, i, t, kwargs):
            state["active"] = i + 1 < active_frac * steps
            return kwargs

        neg = negative if torch.is_tensor(negative) else (self.negative if negative else self.uncond)
        w, h = image.size
        with steer_cross_attention(self.pipe.unet, state):
            out = self.pipe(prompt_embeds=prompt_embeds.half(), negative_prompt_embeds=neg.half(), image=image,
                            mask_image=mask, height=h, width=w, num_inference_steps=steps,
                            guidance_scale=guidance_scale, generator=torch.Generator(self.device).manual_seed(seed),
                            callback_on_step_end=on_step_end).images[0]
        alpha = Image.fromarray(np.maximum(np.asarray(mask.filter(ImageFilter.GaussianBlur(feather_px))),
                                           np.asarray(keep)))
        return Image.composite(out, image, alpha)

    def generate(self, plan, prompt_embeds, seeds=(0,), steps=50, guidance_scale=7.5, negative=True,
                 source_guidance=9.0, method="steer"):
        """steer: one pass over unseen + location, attention steered to the box. mask: pass 1 background in the
        unseen pixels (no audio, steered away from people and animals), pass 2 the source in its location."""
        if method == "steer":
            region = Image.fromarray(np.maximum(np.asarray(plan["inpaint_mask"]), np.asarray(plan["location"])))
            keep = Image.fromarray(np.maximum(np.asarray(plan["hole"]), np.asarray(plan["location"])))
            heat = self.heatmap(plan["box"], plan["warped"].size[0], self.device)[0, 0]
            return [self.fill_steered(plan["warped"], region, keep, prompt_embeds, heat, s, steps, guidance_scale,
                                      negative) for s in seeds]
        has_hole = np.asarray(plan["hole"]).any()
        bg_negative = self.background_negative if negative else False
        images = []
        for s in seeds:
            bg = (self.fill(plan["warped"], plan["inpaint_mask"], plan["hole"], None, s, steps, guidance_scale,
                            bg_negative) if has_hole else plan["warped"])
            images.append(self.fill(bg, plan["location"], plan["location"], prompt_embeds, s, steps, source_guidance,
                                    negative))
        return images


def compass(plan, label, size=256):
    """Top-down sketch: input view (grey), new view (blue), source direction and distance (red)."""
    im = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(im)
    c, R = size / 2, size * 0.42

    def ray(angle_deg, length):  # angle positive = left; forward = up
        a = math.radians(angle_deg)
        return c - length * math.sin(a), c - length * math.cos(a)

    d.ellipse([c - R, c - R, c + R, c + R], outline=(200, 200, 200))
    for yaw, col in ((0.0, (150, 150, 150)), (plan["yaw_deg"], (42, 120, 214))):
        half = plan["half_fov_deg"]
        d.polygon([(c, c), ray(yaw + half, R), ray(yaw - half, R)], outline=col)
    r = min(R, R * plan["distance_m"] / 5.0)
    d.line([(c, c), ray(plan["az_deg"], r)], fill=(227, 73, 72), width=3)
    x, y = ray(plan["az_deg"], r)
    d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(227, 73, 72))
    d.text((6, 4), f"az {plan['az_deg']:+.0f} deg, el {plan['el_deg']:+.0f}", fill="black")
    d.text((6, 18), f"{plan['distance_m']:.1f} m, {label}", fill="black")
    d.text((6, size - 16), f"turn {plan['yaw_deg']:+.0f} deg (blue)", fill=(42, 120, 214))
    return im
