# Sample outputs: look-around with audio

Sample results of the project2 pipeline (depth → MegaScenes-style view warp → audio-conditioned inpainting) on 6 inputs, with the two best camera moves from the pose study (`../novel_view_eval/POSE_STUDY.md`).

- **Inputs:** the sofa photo (`examples/ecohubsofa.jpeg`, a phone photo, center-cropped) and 5 frames from the scene-classification dataset that the model never saw in training: restaurant3-43, store3-160, village1-154, forest662 and classroom5-512, all under `data/scene-classification-images-and-audio/images/`.
- **Poses:**
  - `yawp20`: +20° MegaScenes orbit.
  - `yawm15_back`: −15° with the camera moving back (`--tz 0.2588`) instead of forward.
- **Sounds:** AVSBench S4 test clips: cat `0UN02RkbG_U`, dog `-AWooU1PDiM`, baby `MrIfjRInMLU`, guitar `-3L1rzGAD_o`. Every row also has a no-audio baseline.
- **Seeds:** 0 and 1, with 50 DDIM steps and guidance 7.5. Within a scene and pose, every sound uses the same seeds and the same mask, so only the conditioning differs.

## Files

- `highlights.png`: one row per scene at +20°, seed 0 for every image, so nothing is cherry-picked. Columns: input, no audio, cat, dog, baby, guitar.
- `<scene>_sheet.png`: everything for one scene. The header shows the input, the depth, and for each pose the warped view and the unseen mask. Then there is one row per sound, with both seeds for both poses.
- `<scene>/`: the individual images: `input`, `depth`, `<pose>_warped`, `<pose>_mask` (white = never seen), `<pose>_noaudio_<seed>` and `<pose>_<sound>_<seed>`.

## What we see (by eye)

- **+20°:** with seed 0 (`highlights.png`) the sound's object appears in about 20 of 24 images, and never without audio. Misses: cat on the sofa (garbled text instead), guitar in the village, and baby in the forest and village. Without audio the model paints background, people or furniture.
- **−15° moving back:** the revealed area is mostly the floor and the side of the frame, and audio control is weaker. On the sofa, dog and baby appear for both seeds; cat and guitar don't.
- **Artifacts:**
  - Garbled text or watermarks often appear in the generated region, more often with audio. The adapter was trained on S4 YouTube frames, many of which have overlays.
  - Objects sometimes merge into a person who is already in the scene.
  - Thin gaps next to a foreground object are usually filled with more of that object.

## Reproduce

Every image comes from the same calls as `scripts/novel_view_audio_inpaint.py` with default flags. One combination was checked against the script and matched pixel for pixel. For example:

```bash
W=data/avsbench_data_processed/Single-source/s4_data/audio_wav/test
python scripts/novel_view_audio_inpaint.py --image examples/ecohubsofa.jpeg --wav $W/dog_barking/-AWooU1PDiM.wav \
    --yaw-deg 20 --no-audio-baseline --out /tmp/sofa_p20_dog
python scripts/novel_view_audio_inpaint.py --image examples/ecohubsofa.jpeg --wav $W/dog_barking/-AWooU1PDiM.wav \
    --yaw-deg -15 --tz 0.2588 --no-audio-baseline --out /tmp/sofa_m15_dog
```

The script names its outputs `_audio_<i>` and `_noaudio_<i>`; here they are renamed `<pose>_<sound>_<seed>` and `<pose>_noaudio_<seed>`.
