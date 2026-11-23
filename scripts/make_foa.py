"""Place a mono sound in a virtual room: mono WAV + direction + distance -> 4-channel FOA file (ACN/SN3D).

Direction: azimuth counter-clockwise from straight ahead of the camera (positive = left), elevation up.
The room model (direct path + diffuse reverberation, critical distance 2 m, RT60 0.5 s) is the one
models/spatial_audio.py inverts to recover the distance.

Usage (from the project root):
    python scripts/make_foa.py --wav dog.wav --az 65 --el 0 --distance 2.5 --out dog_left.wav
"""

import argparse
import sys
from pathlib import Path

import librosa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.spatial_audio import FOA_SR, analyse_foa, estimate_distance, save_foa, synthesize_foa  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav", required=True, help="mono (or stereo, mixed down) sound")
    p.add_argument("--az", type=float, required=True, help="degrees, positive = left of the camera")
    p.add_argument("--el", type=float, default=0.0, help="degrees, positive = up")
    p.add_argument("--distance", type=float, default=2.0, help="metres")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    mono, _ = librosa.load(args.wav, sr=FOA_SR, mono=True)
    foa = synthesize_foa(mono, args.az, args.el, args.distance, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_foa(args.out, foa)
    est = analyse_foa(foa)
    print(f"{args.out}: 4 channels, {foa.shape[1] / FOA_SR:.1f} s | placed at az {args.az:+.0f}, el {args.el:+.0f}, "
          f"{args.distance:.1f} m | reads back as az {est['az_deg']:+.1f}, el {est['el_deg']:+.1f}, "
          f"{estimate_distance(est['diffuseness']):.1f} m")


if __name__ == "__main__":
    main()
