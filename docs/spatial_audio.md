# Spatial audio: drawing the right object in the right place

**One line:** with first-order ambisonics (FOA, 4 channels) instead of mono, the sound tells the model *where*
the source is and how far away, not only *what* it is. The camera turns toward the sound, and the object is drawn
at the projected direction, at a size set by its distance, behind nearer surfaces. Two versions: a training-free
one (geometry + cross-attention steering) and a trained one (a heatmap input + LoRA in the inpainting U-Net).

**Headline:** with the sound 65° outside the view, the object appears at the sound's direction in 80.4% of 56
cases with spatial audio against 39.3% with mono (+41.1 points, 95% CI [+26.8, +55.4]).

```
FOA [W, Y, Z, X] ─ intensity vector ─► direction (az, el)     ─┐
                 ─ diffuseness ──────► distance                │
                 ─ beam toward (az, el) ─► CLAP ─► AudioAdapter ─► "a photo of a ✱×8"   what
photo ─ Depth Anything V2 ─► depth                             │
direction ─► camera turn toward the source ─► warp ─► unseen pixels
source point ─► projected (u, v, depth) ─► class size prior ─► box ◄┘
training-free: one pass over the unseen pixels; the audio tokens' cross-attention is steered to the box
trained:       one pass over the unseen pixels with a heatmap at the box (10th U-Net input + LoRA)
```

## 1. Reading FOA (`models/spatial_audio.py`)

- **Format:** ACN channel order [W, Y, Z, X], SN3D normalisation (as in STARSS23); `--fuma` converts FuMa files.
  Azimuth is counter-clockwise from straight ahead (positive = left), elevation positive = up; the microphone
  sits at the camera and faces the same way.
- **Direction:** active intensity I = Re{W* · [X, Y, Z]} per time-frequency bin. Only bins where the direct sound
  dominates are used (louder than the median and with the most aligned intensity, |I| / E in the top 20%), and
  every bin counts equally (unit vectors), because real sounds put most energy in a few low bins where a room's
  response swings most.
- **Distance:** diffuseness ψ = 1 − |Σₜ I| / Σₜ E per frequency, median over frequencies (DirAC). For a direct path
  plus a diffuse reverberant field, ψ = E_rev / (E_dir + E_rev), so the direct-to-reverberant ratio is
  (1 − ψ) / ψ; in a room it falls as (r_c / r)² with the critical distance r_c (2 m here), so
  r = r_c √(ψ / (1 − ψ)).
- **What:** CLAP hears a *hypercardioid beam* steered to the estimated direction, 0.25 W + 0.75 d·[X, Y, Z], not
  the omnidirectional W. The beam keeps the direct sound and cuts diffuse reverberation by about 6 dB.
- **Synthetic FOA** (`synthesize_foa`, `scripts/make_foa.py`): a mono sound as a plane wave at (az, el) with gain
  1 / r, plus a diffuse reverberant tail (exponentially decaying noise, RT60 0.5 s, independent per channel,
  X, Y, Z at 1/3 of W's energy) scaled so DRR = (r_c / r)². We have no real FOA recordings yet; this lets any S4
  clip or WAV be placed anywhere.

Measured on the synthetic room (S4 test clips, 35 placements each, 0.7–5 m):

| sound | direction error, mean (max) | distance error, mean (max) |
|---|---|---|
| dog | 3.3° (14.5°) | 12% (33%) |
| cat | 3.5° (16.7°) | 13% (33%) |
| baby | 3.3° (12.9°) | 10% (33%) |
| guitar | 3.7° (14.0°) | 16% (41%) |

Reverberation also hurts *what*: CLAP zero-shot accuracy over the 23 S4 classes (46 test clips, random
directions):

| signal | 2 m | 3 m | 4 m |
|---|---|---|---|
| clean mono (no room) | 91.3% | 91.3% | 91.3% |
| W channel (one omnidirectional mic) | 73.9% | 71.7% | 69.6% |
| **hypercardioid beam to the estimated direction** | **82.6%** | **76.1%** | **78.3%** |
| beam + diffuseness time-frequency mask (rejected) | 80.4% | 69.6% | 63.0% |

The time-frequency mask (DirAC-style dereverberation) made things worse: its gating artefacts hurt CLAP more than
the reverberation it removes, so only the beam is used.

## 2. Training-free version (`models/spatial_pipeline.py`, `scripts/spatial_audio_inpaint.py`)

1. **Listen:** direction, diffuseness → distance, beam → CLAP → AudioAdapter → prompt; CLAP zero-shot over the 23
   S4 classes (text "the sound of …") gives the class for the size prior (91% top-1 on clean S4 test clips).
2. **Face the source:** if the source is outside the view (more than half the field of view minus 25° off-axis),
   the camera turns toward it along the MegaScenes orbit, up to 30°, stepping back rather than forward (the pose
   study's better variant).
3. **Warp** the photo to the new pose (Depth Anything V2 depth, stretch hole rule, as in the look-around pipeline).
4. **Place the source:** the point r·d in the input camera is projected into the new view → (u, v, z). Box height
   = f · H_class / z (H_class: a hand-set typical height per class, e.g. dog 0.6 m, horse 1.7 m), floored at 40% of
   the image because Stable Diffusion cannot draw an object in a few dozen pixels (a distant source through the
   MegaScenes 116° lens).
5. **Occlusion:** location = box ∧ (unseen ∨ scene depth > z). Nearer surfaces stay in front of the source.
6. **Fill (default, `--method steer`, `models/attention_steering.py`):** one pass over the unseen pixels with the
   audio prompt (guidance 9). In every cross-attention layer, the attention of each pixel to the audio tokens (keys
   from position 5 on, after "a photo of a") is multiplied by w(x) = (0.1 + 0.9 h(x)) (1 + h(x)), with h the
   location heatmap resized to the layer, then renormalised. Only the conditional half of the guidance batch is
   steered, and only for the first 50% of the steps, when the layout is decided; the rest is left alone so the
   object blends in. Inspired by training-free layout control (Chen et al., WACV 2024).
7. **Fill with masks (`--method mask`, rejected):** pass 1 inpaints the unseen pixels with no audio and the
   negative prompt "person, people, animal, text, watermark…" (background only); pass 2 inpaints the location box
   with the audio prompt. The box is often too small or too oddly shaped for Stable Diffusion to draw a whole
   object: the object was found in only 62% of the evaluation clips, against 90% with steering.

## 3. Trained version (`models/location_unet.py`, `dataloaders/avs_s4_location.py`, `scripts/train_location.py`)

- **Model:** the SD-1.5 inpainting U-Net gets a 10th input channel, a location heatmap (Gaussian at the box, std =
  half its size, at 64×64), with zero-initialised weights, plus rank-8 LoRA on to_q / to_k / to_v / to_out of
  every attention layer. With `--multiscale` (the kept model, `runs/loc_v2b_multiscale`) the heatmap is also
  added after every down block through zero-initialised 1×1 convolutions. 1.6M trainable parameters; everything
  else is frozen, including the audio adapter `runs/s4_v1`. At initialisation the model is exactly the original
  (unit test, both variants).
- **Data:** S4 train, frame 1. The hole is a box 1.6–3× the sounding object's box placed at random around it
  (50%), the whole image (25%) or the object alone (25%). The heatmap sits at the object's box centre, jittered by
  31 px (about 4° of direction error) and ±25% in size (the measured errors of section 1); 10% of samples get no
  heatmap.
- **Loss:** the mask-weighted noise-prediction MSE of Approach 1, with extra weight on the object:
  w = 1 + 2·hole + 2·object.
- **Training (v2):** timesteps drawn with a cubic bias toward high noise (as in T2I-Adapter), where the layout is
  decided; AdamW, lr 1e-3 for the heatmap weights and 1e-4 for LoRA, cosine with 500 warm-up steps, batch 4 × 2,
  10k steps, one RTX A5000 (0.88 steps/s, 5.4 GB).
- **What failed first (v1):** uniform timesteps and a single lr of 1e-4. After 2.3k steps the heatmap weights had
  barely moved from zero (norm 0.1, against 2–3 for the image channels) and the validation error with the correct
  heatmap was no lower than with none (0.13171 vs 0.13170), so the run was stopped. Most uniformly drawn steps are
  at noise levels where the object's position is already fixed, so the heatmap carries little signal. v2 ends with
  correct < mirrored and none (0.10006 vs 0.10012 and 0.10014).
- **Inference:** one pass over the unseen pixels; the heatmap goes into both classifier-free-guidance branches
  (it is *where*, not *what*).

## 4. Evaluation

Every method gets the same clips, holes, seeds and steps. Place is judged with OWLv2 (open-vocabulary detection of
the clip's class inside the hole), the class with CLIP ViT-L/14 zero-shot. Differences between methods are paired
(same cases) with 95% bootstrap intervals (`scripts/spatial_stats.py`).

### 4.1 In place (`scripts/eval_spatial.py`, `outputs/spatial_eval*`)

173 S4 test clips whose sounding object is at most a third of the frame wide. The hole is 3× the object's width
with the object at one end, so the model must choose a side; the clip's audio is rendered as FOA at the object's
true direction and distance. **Centre hit:** the detected box's centre lies in the real object's box; **IoU hit:**
IoU ≥ 0.3 (place and size).

| conditioning | found | centre hit | IoU hit | CLIP top-1 |
|---|---|---|---|---|
| no audio | 31.8% | 8.1% | 6.4% | 30.6% |
| mono audio (W channel) | 86.1% | 34.7% | 25.4% | 78.6% |
| training-free, masks (rejected) | 62.4% | 28.9% | 23.1% | 56.1% |
| **training-free, attention steering** | 89.6% | 39.9% | 32.4% | 82.7% |
| trained, input channel (`loc_v2a_convin`) | 90.8% | **45.7%** | 36.4% | 82.1% |
| trained, multiscale (`loc_v2b_multiscale`) | 90.2% | 43.4% | **40.5%** | **89.0%** |

Paired differences (points, 95% CI):

| comparison | centre hit | IoU hit | CLIP top-1 |
|---|---|---|---|
| steering − mono | +5.2 [−1.7, +12.1] | +6.9 [+1.2, +12.7] | +4.0 [−1.2, +9.8] |
| trained multiscale − mono | +8.7 [+0.6, +16.8] | +15.0 [+6.9, +23.1] | +10.4 [+4.0, +16.8] |
| trained multiscale − steering | +3.5 [−4.6, +12.1] | +8.1 [+0.6, +16.2] | +6.4 [+1.2, +12.1] |

In place the trained model is best; steering beats mono on place and size (IoU hit) but its centre-hit gain is
within noise.

### 4.2 Around the corner (`scripts/eval_spatial_lookaround.py`, `outputs/spatial_lookaround`)

14 scenes (8 S4 test frames, 5 Kaggle photos, our sofa photo) × the sound 65° left or right, 2.5 m away, outside
the view × 2 sounds = 56 cases. Mono audio cannot tell where the sound is, so it turns the camera a fixed +20°
(left); the spatial methods turn toward the estimated direction. **Hit:** OWLv2 finds the object on the sound's
side of the new view.

| conditioning | found | hit | hit, sound left | hit, sound right |
|---|---|---|---|---|
| mono audio, fixed turn | 85.7% | 39.3% | 75.0% | 3.6% |
| **spatial, training-free (steering)** | 92.9% | **80.4%** | 82.1% | 78.6% |
| spatial, trained (multiscale) | 92.9% | 69.6% | 64.3% | 75.0% |

Spatial − mono: +41.1 points [+26.8, +55.4]; trained − mono: +30.4 [+12.5, +48.2]; training-free − trained: +10.7
[−1.8, +23.2] (not significant). The trained model only saw S4 frames, not warped wide-angle views, so it is not
better here; the training-free version needs no training and is the default pipeline.

## 5. Limitations

- **No real FOA data yet.** All spatial audio is synthetic (one room model). Real recordings (STARSS23, MRSAudio)
  bring more reverberation types, multiple sources and noise; the direction estimator is standard and should
  carry over, the distance estimate depends on knowing the room's critical distance.
- **Relative depth.** Depth Anything V2 gives relative depth; `--scene-scale-m` (default 2 m per unit) converts it
  to metres for occlusion and size, and is a guess per scene.
- **Size priors** are hand-set per class.
- **S4 objects are mostly large and centred**, so only 173 of 740 test clips leave a location to choose.
