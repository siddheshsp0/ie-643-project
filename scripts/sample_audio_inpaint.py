"""Inference: inpaint the masked region of an image, with a sound deciding what gets drawn there.

The WAV goes through CLAP, the trained adapter and CLIP, and the result is passed to the standard
diffusers inpainting pipeline as prompt_embeds (the empty prompt is the negative prompt).

Usage (from the project root):
    # any audio + image + mask (white = region to repaint; omit --mask to repaint the whole image)
    python scripts/sample_audio_inpaint.py --ckpt runs/s4_v1/last.pt \
        --wav some.wav --image frame.png --mask mask.png --out outputs/demo

    # an AVSBench S4 clip by name: uses its WAV, frame 1 and dilated ground-truth mask
    python scripts/sample_audio_inpaint.py --ckpt runs/s4_v1/last.pt --s4-split test --s4-name <youtube_id>

    # same S4 frame/mask but the audio of another clip (audio swap)
    python scripts/sample_audio_inpaint.py --ckpt runs/s4_v1/last.pt --s4-split test --s4-name <id> --wav other.wav
"""

import argparse
import sys
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloaders.avs_s4 import frame_path, label_text, mask_path, read_split, wav_path  # noqa: E402
from models.audio_adapter import AudioAdapter  # noqa: E402
from models.audio_token_conditioner import DEFAULT_SD_PATH, AudioTokenConditioner  # noqa: E402
from models.clap_feature_extractor import CLAPFeatureExtractor  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="adapter checkpoint, e.g. runs/s4_v1/last.pt")
    p.add_argument("--wav", help="conditioning audio (overrides the S4 clip's audio)")
    p.add_argument("--image", help="image to inpaint")
    p.add_argument("--mask", help="mask image, white = repaint (default: whole image)")
    p.add_argument("--s4-split", choices=["train", "val", "test"])
    p.add_argument("--s4-name", help="S4 clip id: supplies WAV, frame and ground-truth mask")
    p.add_argument("--s4-frame", type=int, default=1, help="frame 1-5 (train has a mask only for frame 1)")
    p.add_argument("--dilate-px", type=int, default=8, help="dilation of S4 masks (training used 8)")
    p.add_argument("--out", default="outputs/sample", help="output path prefix")
    p.add_argument("--num-images", type=int, default=2)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-audio-baseline", action="store_true",
                   help="also generate with an empty prompt (same seeds) for comparison")
    p.add_argument("--sd-path", default=str(DEFAULT_SD_PATH))
    return p.parse_args()


def resolve_inputs(args):
    """Returns (wav, image, mask, caption)."""
    if args.s4_name:
        if not args.s4_split:
            raise SystemExit("--s4-name needs --s4-split")
        rec = next((r for r in read_split(args.s4_split) if r["name"] == args.s4_name), None)
        if rec is None:
            raise SystemExit(f"{args.s4_name} not found in the {args.s4_split} split")
        mpath = mask_path(rec, args.s4_frame)
        if not mpath.is_file():
            raise SystemExit(f"no ground-truth mask for frame {args.s4_frame}: {mpath}")
        image = Image.open(frame_path(rec, args.s4_frame)).convert("RGB").resize((512, 512), Image.BICUBIC)
        mask = Image.open(mpath).convert("L").resize((512, 512), Image.NEAREST)
        if args.dilate_px > 0:
            mask = mask.filter(ImageFilter.MaxFilter(2 * args.dilate_px + 1))
        wav = Path(args.wav) if args.wav else wav_path(rec)
        caption = f"image: {label_text(rec['category'])}  |  audio: {wav.name}"
        return wav, image, mask, caption
    if not (args.wav and args.image):
        raise SystemExit("give --wav and --image, or --s4-split/--s4-name")
    image = Image.open(args.image).convert("RGB").resize((512, 512), Image.BICUBIC)
    if args.mask:
        mask = Image.open(args.mask).convert("L").resize((512, 512), Image.NEAREST)
    else:
        mask = Image.new("L", (512, 512), 255)
    return Path(args.wav), image, mask, f"audio: {Path(args.wav).name}"


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav, image, mask, caption = resolve_inputs(args)

    ckpt = torch.load(args.ckpt, map_location=device)
    print(f"checkpoint {args.ckpt} (step {ckpt['step']})  layers {ckpt['layer_names']}")

    extractor = CLAPFeatureExtractor(layers=ckpt["layer_names"], device=device)
    adapter = AudioAdapter(**ckpt["adapter_config"]).to(device).eval()
    adapter.load_state_dict(ckpt["adapter"])
    conditioner = AudioTokenConditioner(args.sd_path, template=ckpt["template"],
                                        num_tokens=ckpt["adapter_config"]["num_tokens"], device=device)

    with torch.no_grad():
        feats = extractor.extract([str(wav)])
        cond = conditioner(adapter(feats["intermediates"], feats["global_embed"]))  # [1, 77, 768]
        tok = conditioner.tokenizer
        empty = tok("", padding="max_length", max_length=tok.model_max_length, return_tensors="pt").input_ids
        uncond = conditioner.text_encoder(input_ids=empty.to(device)).last_hidden_state
    del extractor
    torch.cuda.empty_cache()

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.sd_path, dtype=torch.float16, variant="fp16", safety_checker=None,
        feature_extractor=None, requires_safety_checker=False).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    def run(prompt_embeds):
        gens = [torch.Generator(device).manual_seed(args.seed + i) for i in range(args.num_images)]
        n = args.num_images
        return pipe(prompt_embeds=prompt_embeds.half().expand(n, -1, -1),
                    negative_prompt_embeds=uncond.half().expand(n, -1, -1),
                    image=image, mask_image=mask, num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale, generator=gens).images

    audio_imgs = run(cond)
    empty_imgs = run(uncond) if args.no_audio_baseline else []

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(audio_imgs):
        im.save(f"{out}_audio_{i}.png")
    for i, im in enumerate(empty_imgs):
        im.save(f"{out}_noaudio_{i}.png")

    # Overview: masked input | original | audio samples | (no-audio samples)
    gray = Image.new("RGB", image.size, (128, 128, 128))
    masked = Image.composite(gray, image, mask.point(lambda v: 255 if v > 127 else 0))
    tiles = [("masked input", masked), ("original", image)]
    tiles += [(f"audio, seed {args.seed + i}", im) for i, im in enumerate(audio_imgs)]
    tiles += [(f"no audio, seed {args.seed + i}", im) for i, im in enumerate(empty_imgs)]
    t = 256
    sheet = Image.new("RGB", (t * len(tiles), t + 40), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 2), caption, fill="black")
    for c, (label, im) in enumerate(tiles):
        draw.text((c * t + 4, 20), label, fill="black")
        sheet.paste(im.resize((t, t), Image.BICUBIC), (c * t, 40))
    sheet.save(f"{out}_overview.png")
    print(f"saved {len(audio_imgs) + len(empty_imgs)} images and {out}_overview.png")


if __name__ == "__main__":
    main()
