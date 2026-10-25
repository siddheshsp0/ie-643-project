"""Look around occluders: warp one image to a new camera pose and let the audio model fill what it never saw.

Depth Anything V2 gives depth, the MegaScenes warp moves the camera (15 deg by default), and the trained
audio adapter + SD inpainting generates the holes. Nothing is trained here. See docs/novel_view_pipeline.md.

Usage (from the project root):
    python scripts/novel_view_audio_inpaint.py --image room.jpg --wav cat.wav --out outputs/novel_view/room_cat

    # orbit the other way (camera moves left, reveals the left side of near objects)
    python scripts/novel_view_audio_inpaint.py --image room.jpg --wav cat.wav --yaw-deg -15

    # an AVSBench S4 frame with its own audio, or another clip's audio via --wav
    python scripts/novel_view_audio_inpaint.py --s4-split test --s4-name <youtube_id> --wav other.wav

Outputs for prefix --out: _input, _depth (+ .npy), _warped, _mask (white = hole), _inpaint_mask,
_audio_<i>, _noaudio_<i> (with --no-audio-baseline) and _overview.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloaders.avs_s4 import frame_path, label_text, read_split, wav_path  # noqa: E402
from models.audio_token_conditioner import DEFAULT_SD_PATH  # noqa: E402
from models.depth_anything_v2 import DEFAULT_CKPT as DEFAULT_DA_CKPT  # noqa: E402
from models.depth_anything_v2 import MODEL_CONFIGS  # noqa: E402
from models.megascenes_warp import MEGASCENES_FOCAL_256, PRUNE_RULES, orbit_extrinsic  # noqa: E402
from models.novel_view_pipeline import (NovelViewAudioInpainter, center_crop_square,  # noqa: E402
                                        colorize_disparity, warp_view)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", help="input view (center-cropped to a square)")
    p.add_argument("--wav", help="conditioning audio (overrides the S4 clip's audio)")
    p.add_argument("--s4-split", choices=["train", "val", "test"])
    p.add_argument("--s4-name", help="S4 clip id: supplies frame and WAV")
    p.add_argument("--s4-frame", type=int, default=1, help="frame 1-5")
    p.add_argument("--ckpt", default=str(ROOT / "runs/s4_v1/last.pt"), help="audio adapter checkpoint")
    p.add_argument("--out", default="outputs/novel_view/sample", help="output path prefix")

    g = p.add_argument_group("camera (MegaScenes orbit, units of normalized depth)")
    g.add_argument("--yaw-deg", type=float, default=15.0,
                   help="rotation about the vertical axis; > 0 turns the camera left while it moves right")
    g.add_argument("--tx", type=float, help="x translation of the extrinsic (default: -yaw/27, the MegaScenes orbit)")
    g.add_argument("--ty", type=float, default=0.0)
    g.add_argument("--tz", type=float, help="z translation of the extrinsic (default: sin(yaw))")
    g.add_argument("--focal-256", type=float, default=MEGASCENES_FOCAL_256,
                   help="focal length in px for a 256 px image (MegaScenes uses 80)")
    g.add_argument("--prune", choices=PRUNE_RULES, default="stretch",
                   help="which mesh faces become holes: 'stretch' = only those torn open by parallax "
                        "(holes = unseen pixels); 'megascenes' = MegaScenes' angle rule, which also "
                        "drops slanted/far surfaces")
    g.add_argument("--max-stretch", type=float, default=4.0, help="'stretch' rule threshold")
    g.add_argument("--prune-angle", type=float, default=70.0, help="'megascenes' rule threshold in degrees")

    g = p.add_argument_group("depth")
    g.add_argument("--da-ckpt", default=str(DEFAULT_DA_CKPT))
    g.add_argument("--da-encoder", default="vitl", choices=sorted(MODEL_CONFIGS))

    g = p.add_argument_group("inpainting")
    g.add_argument("--resolution", type=int, default=512)
    g.add_argument("--dilate-px", type=int, default=8,
                   help="grow the hole mask before inpainting; covers depth-edge pixels, which the renderer "
                        "anti-aliases against black (S4 training used 8)")
    g.add_argument("--num-images", type=int, default=2)
    g.add_argument("--steps", type=int, default=50)
    g.add_argument("--guidance-scale", type=float, default=7.5)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--no-audio-baseline", action="store_true",
                   help="also inpaint with an empty prompt (same seeds) for comparison")
    g.add_argument("--sd-path", default=str(DEFAULT_SD_PATH))
    return p.parse_args()


def resolve_inputs(args):
    """Returns (wav, square full-resolution image, caption)."""
    if args.s4_name:
        if not args.s4_split:
            raise SystemExit("--s4-name needs --s4-split")
        rec = next((r for r in read_split(args.s4_split) if r["name"] == args.s4_name), None)
        if rec is None:
            raise SystemExit(f"{args.s4_name} not found in the {args.s4_split} split")
        image = Image.open(frame_path(rec, args.s4_frame)).convert("RGB")
        wav = Path(args.wav) if args.wav else wav_path(rec)
        caption = f"image: {label_text(rec['category'])}  |  audio: {wav.name}"
        return wav, center_crop_square(image), caption
    if not (args.wav and args.image):
        raise SystemExit("give --wav and --image, or --s4-split/--s4-name")
    image = Image.open(args.image).convert("RGB")
    return Path(args.wav), center_crop_square(image), f"image: {Path(args.image).name}  |  audio: {Path(args.wav).name}"


def main():
    args = parse_args()
    wav, image, caption = resolve_inputs(args)
    res = args.resolution
    src = image.resize((res, res), Image.LANCZOS)
    model = NovelViewAudioInpainter(args.ckpt, args.sd_path, args.da_ckpt, args.da_encoder)

    disparity, depth = model.depth(image, res)
    E = orbit_extrinsic(args.yaw_deg, args.tx, args.ty, args.tz)
    warped, hole, mask = warp_view(src, depth, E, args.focal_256, args.prune, args.max_stretch, args.prune_angle,
                                   args.dilate_px)
    print(f"pose: yaw {args.yaw_deg:+.1f} deg, t = {np.round(E[:3, 3], 3).tolist()}  |  "
          f"holes {100 * (np.asarray(hole) > 127).mean():.1f}% of the view, "
          f"{100 * (np.asarray(mask) > 127).mean():.1f}% inpainted")

    seeds = [args.seed + i for i in range(args.num_images)]
    kw = dict(seeds=seeds, steps=args.steps, guidance_scale=args.guidance_scale, feather_px=max(args.dilate_px / 2, 1))
    audio_imgs = model.inpaint(warped, hole, mask, model.audio_embeds(wav), **kw)
    empty_imgs = model.inpaint(warped, hole, mask, None, **kw) if args.no_audio_baseline else []

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    depth_vis = colorize_disparity(disparity)
    src.save(f"{out}_input.png")
    depth_vis.save(f"{out}_depth.png")
    np.save(f"{out}_depth.npy", depth)
    warped.save(f"{out}_warped.png")
    hole.save(f"{out}_mask.png")
    mask.save(f"{out}_inpaint_mask.png")
    for i, im in enumerate(audio_imgs):
        im.save(f"{out}_audio_{i}.png")
    for i, im in enumerate(empty_imgs):
        im.save(f"{out}_noaudio_{i}.png")

    # Overview: input | depth | warped | hole mask | audio samples | (no-audio samples)
    tiles = [("input view", src), ("depth (DA-V2)", depth_vis), (f"warped, yaw {args.yaw_deg:+g} deg", warped),
             ("hole mask", hole)]
    tiles += [(f"audio, seed {s}", im) for s, im in zip(seeds, audio_imgs)]
    tiles += [(f"no audio, seed {s}", im) for s, im in zip(seeds, empty_imgs)]
    t = 256
    sheet = Image.new("RGB", (t * len(tiles), t + 40), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 2), caption, fill="black")
    for c, (label, im) in enumerate(tiles):
        draw.text((c * t + 4, 20), label, fill="black")
        sheet.paste(im.convert("RGB").resize((t, t), Image.BICUBIC), (c * t, 40))
    sheet.save(f"{out}_overview.png")
    print(f"saved {len(audio_imgs) + len(empty_imgs)} images and {out}_overview.png")


if __name__ == "__main__":
    main()
