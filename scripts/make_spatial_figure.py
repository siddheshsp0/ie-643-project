"""Qualitative figure for the around-the-corner evaluation: input | mono (+20 deg) | spatial, one row per case.

    python scripts/make_spatial_figure.py --tags village_dog_right forest_cat_right --out figure.png
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.novel_view_pipeline import center_crop_square  # noqa: E402
from scripts.eval_spatial_lookaround import scenes  # noqa: E402


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--src", default=str(ROOT / "outputs/spatial_lookaround"))
    p.add_argument("--tile", type=int, default=320)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    src, t, gap, head = Path(args.src), args.tile, 8, 40
    paths = dict(scenes())
    cols = ["input view", "mono audio: fixed +20° turn", "spatial: turns toward sound"]
    sheet = Image.new("RGB", (3 * t + 2 * gap, head + len(args.tags) * (t + 34)), "white")
    d = ImageDraw.Draw(sheet)
    for k, c in enumerate(cols):
        d.text((k * (t + gap) + 6, 10), c, fill="black", font=font(19))
    for r, tag in enumerate(args.tags):
        m = json.loads((src / "meta" / f"{tag}.json").read_text())
        y = head + r * (t + 34)
        tiles = [center_crop_square(Image.open(paths[m["scene"]]).convert("RGB")),
                 Image.open(src / "mono" / f"{tag}.png"), Image.open(src / "spatial" / f"{tag}.png")]
        for k, im in enumerate(tiles):
            sheet.paste(im.convert("RGB").resize((t, t), Image.BICUBIC), (k * (t + gap), y))
        side = "left" if m["az"] > 0 else "right"
        d.text((6, y + t + 6), f"{m['sound']} sound {abs(m['az']):.0f}° to the {side}, outside the view: "
                               f"heard at {m['est_az']:+.0f}°, camera turned {m['yaw_spatial']:+.0f}°",
               fill="black", font=font(16))
    sheet.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
