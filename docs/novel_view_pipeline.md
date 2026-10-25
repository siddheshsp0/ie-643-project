# Look Around Occluders: Depth → MegaScenes Warp → Audio-Conditioned Inpainting

**One line:** estimate depth for a single image, move the camera MegaScenes-style (15° yaw plus translation), keep only the pixels the input actually saw, and let the audio-conditioned inpainting model (`runs/s4_v1/last.pt`) generate everything that was hidden. A barking dog or a meowing cat can then appear in the part of the scene the original view could not see. Nothing is trained; all models are frozen.

```
 image ──Depth Anything V2 (ViT-L)──► disparity ─► depth = 1/disp, ÷ 20th percentile   (MegaScenes)
                                                        │
                    mesh (1 vertex / pixel) ◄───────────┘
                        │  pose: R_y(15°), t = (−0.556, 0, 0.259)                        (MegaScenes orbit)
                        ▼  pyrender, fx = 80 px @ 256 px                                 (MegaScenes renderer)
             warped view (unseen pixels black) + hole mask
                        │
 mono WAV ──CLAP──► AudioAdapter ──► CLIP "a photo of a * x 8" ──► SD-1.5 inpainting fills the holes
```

## Run

```bash

python scripts/novel_view_audio_inpaint.py --image room.jpg --wav cat.wav \
    --out outputs/novel_view/room_cat --no-audio-baseline

python -m unittest tests.test_megascenes_warp -v        # geometry tests (GPU/EGL)
```

Useful flags: `--yaw-deg -15` orbits the other way. `--tx/--ty/--tz` override the translation. `--prune megascenes` switches to MegaScenes' exact hole rule. `--dilate-px` grows the inpainting mask (default 8). `--s4-split/--s4-name` load an AVSBench frame and its audio.

Outputs for prefix `P`: `P_input`, `P_depth.png/.npy`, `P_warped` (holes black), `P_mask` (white = never seen), `P_inpaint_mask` (dilated), `P_audio_<i>`, `P_noaudio_<i>` and `P_overview`.

The models live in `models/novel_view_pipeline.py` (`NovelViewAudioInpainter`, loaded once), so grids are cheap.

### Evaluation grid

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_novel_view.py generate --shard 0 --num-shards 2 &
CUDA_VISIBLE_DEVICES=1 python scripts/eval_novel_view.py generate --shard 1 --num-shards 2
python scripts/eval_novel_view.py report      # -> outputs/novel_view_eval/v1/README.md
```

Pose variants take `--pose no-forward` (−yaw also moves back), `--pose rotation-only`, `--focal-256 F` and `--yaws`. `scripts/compare_pose_study.py v1 <runs...>` compares runs on the seeds they all share and writes `outputs/novel_view_eval/POSE_STUDY.md`. That study found that removing the −15° forward move adds about 19 points of audio top-1, and +20° adds about 20 points.

The grid is 8 S4 test scenes × ±15° × {no audio, cat, dog, baby, guitar} × 3 seeds. Audio and no-audio images share seeds and masks. `report` scores **only the generated pixels** with CLIP ViT-L/14 zero-shot over the 23 S4 classes (`models/clip-vit-large-patch14`, used only for evaluation). It writes `metrics.csv`, `summary.json`, annotated contact sheets and a markdown report.

## What comes from MegaScenes, and the one deviation

| Step | Source |
|---|---|
| Depth: DA disparity → bicubic resize → `1/max(d, 1e-3)` → ÷ 20th percentile → clip [0, 100] | `video_script.py`, `depth_helpers.py` (DA v1 there, DA **V2** ViT-L here) |
| Intrinsics `fx = fy = 80` at 256 px, principal point at the center (scaled to 512) | `video_script.py` |
| Pose: `get_ccw_poses`/`get_cw_poses` orbit, `x = −θ/27°`, `z = sin θ`; ±15° reproduces their steps exactly (unit-tested against their code) | `evalhelpers.py` |
| Mesh + render: `unproject_depth`, `render_multiviews` (loaded unmodified) | `util_3dphoto.py` |

**Deviation: which mesh faces become holes.** MegaScenes drops every face seen at more than 70° from the camera. With inverted relative disparity, far and slanted surfaces are very steep, so **19–25% of the input image goes missing even with no camera motion** (walls and floors turn into streaks). That contradicts "only unseen pixels are empty", and it gives the audio model random streaks to paint the sound source into. The default `--prune stretch` drops only faces that **tear open** in the new view: their longest edge grows more than 4× beyond the local zoom, which happens where a face bridges a depth discontinuity. Verified properties:
- identity pose → no holes, input reproduced;
- the region behind a near object becomes a hole;
- on a continuous surface the holes are **exactly** the pixels whose rays leave the original frame (analytic ray–plane test).

`--prune megascenes` restores their rule. The mesh is also built from a 1 px edge-padded image, because MegaScenes' pixel-center vertices otherwise leave a false 1 px hole along two borders.

Notes:
- MegaScenes' ±15° poses are **not mirror images**. At +15° the camera moves right and slightly back and turns left. At −15° it moves left and **forward by 0.39** (their `tz = sin θ < 0`) and turns right, so the view zooms in.
- pyrender renders with 4× MSAA. Pixels on hole edges are anti-aliased against black, which is one reason the mask is dilated.

## Observed behaviour (`outputs/novel_view/`)

- `desk_cat`, `desk_dog`: a 15° orbit reveals a wide region to the left of the hand. With cat or dog audio, a **cat or dog appears there in both seeds**. The no-audio baseline (same seeds, same mask) paints a person or arm instead.
- `doorway_dog_cat`: the hole directly behind the dog is a thin, dog-shaped sliver attached to the dog. SD **extends the dog** into it with or without audio, so no cat appears. Audio steers large revealed regions much more than slivers hugging a foreground object.
- SD-1.5 inpainting sometimes writes garbled text into large bottom-edge holes, in both conditions.
- With MegaScenes' wide FoV (≈116° horizontal), a 15° orbit leaves ~30–50% of the view to generate.

## Environment and provenance

- `venvs/project2` is an overlay venv: a `.pth` file exposes the original `venvs/ie643_project` packages **read-only**, and only `pyrender 0.1.45`, `trimesh`, `PyOpenGL 3.1.9` (newer than pyrender's pin, needed for EGL on Python 3.13) and `icecream` are installed into it. The original venv is untouched.
- `third_party/Depth-Anything-V2` @ `a561b84`: code Apache-2.0. **The ViT-L weights (`models/depth-anything-v2/depth_anything_v2_vitl.pth`, sha256 `a7ea19fa…`) are CC-BY-NC-4.0, non-commercial only.** ViT-S (Apache-2.0) works via `--da-encoder vits --da-ckpt <depth_anything_v2_vits.pth>` (not downloaded).
- `third_party/megascenes_nvs` @ `0de3b00`: no license file in the repo. Only the warping/pose code is used. MegaScenes' NVS diffusion checkpoints were **not** downloaded, because the audio inpainting model replaces their generator.
