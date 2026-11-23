"""Spatial-audio look-around: the FOA recording decides what appears, where, and how big (training-free).

The camera turns toward the sound (MegaScenes orbit, at most --max-yaw degrees), the region the photo never
saw is filled as background, and the source is drawn only at its projected direction, at a size set by its
class and estimated distance, behind any nearer surface. See models/spatial_pipeline.py.

Usage (from the project root):
    # a real FOA recording (4 channels, ACN/SN3D; add --fuma for FuMa files)
    python scripts/spatial_audio_inpaint.py --image room.jpg --foa scene.wav --out outputs/spatial/room

    # a mono sound placed at a direction (writes <out>_foa.wav), compared with the mono model and no audio
    python scripts/spatial_audio_inpaint.py --image room.jpg --wav dog.wav --az 65 --distance 2.5 --compare \
        --out outputs/spatial/room_dog_left

Outputs (prefix --out): _input, _depth, _warped, _mask (unseen), _location, _spatial_<seed>, with --compare
_mono_<seed> (the mono look-around at a fixed +20 deg) and _noaudio_<seed>, _plan.json and _overview.
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.megascenes_warp import MEGASCENES_FOCAL_256, orbit_extrinsic  # noqa: E402
from models.novel_view_pipeline import colorize_disparity, warp_view  # noqa: E402
from models.spatial_audio import FOA_SR, load_foa, save_foa, synthesize_foa  # noqa: E402
from models.spatial_pipeline import SpatialAudioInpainter, compass  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--foa", help="4-channel FOA recording (ACN/SN3D)")
    p.add_argument("--fuma", action="store_true", help="the FOA file is FuMa ordered")
    p.add_argument("--wav", help="mono sound to place with --az/--el/--distance instead of --foa")
    p.add_argument("--az", type=float, help="degrees, positive = left of the camera")
    p.add_argument("--el", type=float, default=0.0)
    p.add_argument("--distance", type=float, default=2.0, help="metres (for --wav)")
    p.add_argument("--ckpt", default=str(ROOT / "runs/s4_v1/last.pt"))
    p.add_argument("--location-ckpt", help="trained location model (runs/loc_v2b_multiscale/last.pt): adds the trained result")
    p.add_argument("--out", default="outputs/spatial/sample")
    g = p.add_argument_group("geometry")
    g.add_argument("--yaw-deg", type=float, help="force the camera turn (default: face the source)")
    g.add_argument("--max-yaw", type=float, default=30.0)
    g.add_argument("--margin-deg", type=float, default=25.0, help="how far inside the frame the source must land")
    g.add_argument("--scene-scale-m", type=float, default=2.0, help="metres per unit of relative depth")
    g.add_argument("--focal-256", type=float, default=MEGASCENES_FOCAL_256)
    g.add_argument("--size-m", type=float, help="override the class height prior (metres)")
    g = p.add_argument_group("generation")
    g.add_argument("--resolution", type=int, default=512)
    g.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    g.add_argument("--steps", type=int, default=50)
    g.add_argument("--guidance-scale", type=float, default=7.5)
    g.add_argument("--no-negative", action="store_true", help="empty negative prompt instead of 'text, watermark…'")
    g.add_argument("--compare", action="store_true", help="also run the mono look-around (+20 deg) and no audio")
    g.add_argument("--method", choices=["steer", "mask"], default="steer",
                   help="steer: attention steered to the source (default); mask: background pass + source pass")
    return p.parse_args()


def load_inputs(args, out):
    if args.foa:
        return load_foa(args.foa, fuma=args.fuma)
    if not (args.wav and args.az is not None):
        raise SystemExit("give --foa, or --wav with --az")
    mono, _ = librosa.load(args.wav, sr=FOA_SR, mono=True)
    foa = synthesize_foa(mono, args.az, args.el, args.distance)
    save_foa(f"{out}_foa.wav", foa)
    return foa, FOA_SR


def overlay_box(image, box, location):
    im = image.convert("RGB").copy()
    tint = Image.new("RGB", im.size, (227, 73, 72))
    im = Image.composite(Image.blend(im, tint, 0.4), im, location)
    ImageDraw.Draw(im).rectangle(box, outline=(227, 73, 72), width=3)
    return im


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    foa, sr = load_inputs(args, out)
    model = SpatialAudioInpainter(args.ckpt)
    prompt, cls, p_cls, _ = model.listen(foa, sr)
    image = Image.open(args.image)
    plan = model.plan(image, foa, sr, args.resolution, args.yaw_deg, args.max_yaw, args.margin_deg, args.focal_256,
                      args.scene_scale_m, cls, args.size_m)
    print(f"sound: {cls} (CLAP {p_cls:.2f}) | az {plan['az_deg']:+.1f}, el {plan['el_deg']:+.1f}, "
          f"{plan['distance_m']:.1f} m | camera turn {plan['yaw_deg']:+.1f} deg | source at "
          f"({plan['u']:.0f}, {plan['v']:.0f}){'' if plan['visible'] else ' (clamped to the edge)'}"
          f"{' | behind nearer geometry' if plan['occluded'] else ''}")
    kw = dict(steps=args.steps, guidance_scale=args.guidance_scale, negative=not args.no_negative)
    spatial = model.generate(plan, prompt, args.seeds, method=args.method, **kw)

    tiles = [("input", plan["src"]), ("depth (DA-V2)", colorize_disparity(plan["disparity"])),
             ("where the sound is", compass(plan, cls)),
             (f"turned {plan['yaw_deg']:+.0f} deg, unseen black", plan["warped"]),
             ("source location", overlay_box(plan["warped"], plan["box"], plan["location"]))]
    tiles += [(f"spatial audio, seed {s}", im) for s, im in zip(args.seeds, spatial)]
    for s, im in zip(args.seeds, spatial):
        im.save(f"{out}_spatial_{s}.png")
    if args.location_ckpt:
        model.load_location(args.location_ckpt)
        trained = model.generate_trained(plan, prompt, args.seeds, args.steps, args.guidance_scale,
                                         not args.no_negative)
        tiles += [(f"spatial, trained, seed {s}", im) for s, im in zip(args.seeds, trained)]
        for s, im in zip(args.seeds, trained):
            im.save(f"{out}_trained_{s}.png")

    if args.compare:  # the mono look-around of project2: fixed +20 deg, the whole unseen region gets the prompt
        mono_view, mono_hole, mono_mask = warp_view(plan["src"], plan["depth"], orbit_extrinsic(20.0), args.focal_256)
        mono = model.inpaint(mono_view, mono_hole, mono_mask, prompt, seeds=args.seeds, steps=args.steps,
                             guidance_scale=args.guidance_scale)
        none = [model.fill(plan["warped"], plan["inpaint_mask"], plan["hole"], None, s, **kw)
                if np.asarray(plan["hole"]).any() else plan["warped"] for s in args.seeds]
        for s, a, b in zip(args.seeds, mono, none):
            a.save(f"{out}_mono_{s}.png")
            b.save(f"{out}_noaudio_{s}.png")
        tiles += [(f"mono audio, +20 deg, seed {s}", im) for s, im in zip(args.seeds, mono)]
        tiles += [(f"no audio, seed {s}", im) for s, im in zip(args.seeds, none)]

    plan["src"].save(f"{out}_input.png")
    colorize_disparity(plan["disparity"]).save(f"{out}_depth.png")
    plan["warped"].save(f"{out}_warped.png")
    plan["hole"].save(f"{out}_mask.png")
    plan["location"].save(f"{out}_location.png")
    meta = {k: plan[k] for k in ("az_deg", "el_deg", "diffuseness", "distance_m", "yaw_deg", "u", "v", "z", "box",
                                 "visible", "occluded")}
    meta.update(source_class=cls, clap_prob=p_cls, image=args.image, foa=args.foa or f"{out}_foa.wav",
                placed=None if args.foa else {"az": args.az, "el": args.el, "distance_m": args.distance})
    Path(f"{out}_plan.json").write_text(json.dumps(meta, indent=2, default=float))

    t = 256
    sheet = Image.new("RGB", (t * len(tiles), t + 24), "white")
    d = ImageDraw.Draw(sheet)
    for c, (label, im) in enumerate(tiles):
        d.text((c * t + 4, 4), label, fill="black")
        sheet.paste(im.convert("RGB").resize((t, t), Image.BICUBIC), (c * t, 24))
    sheet.save(f"{out}_overview.png")
    print(f"saved {out}_overview.png")


if __name__ == "__main__":
    main()
