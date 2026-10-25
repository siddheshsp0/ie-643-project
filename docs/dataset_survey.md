# Datasets found:
## Simplified version:
- VGGSound (links to yt videos seem corrupted)
- AVSBench S4 and MS3 (got annotated and processed data from the prof) need to explore new processed data
- MS COCO + Captions (unexplored)
- https://jinxiang-liu.github.io/anno-free-AVS/ (found this in a video), object mask + object audio

- AVE (optional, low priority)

## Original version:
- MRSAudio () to explore
- STARSS23 (360 view) unexplored




Dataset Research Status
========================

1. VGGSound
------------
Status: Known, not obtained
Priority: Medium

- Large audio-visual dataset: 200K+ clips, 310+ classes, 550+ hours.
- Each clip is a real-world video with synchronized audio.
- The sound source is usually visible in the video.
- Useful for learning general audio <-> visual relationships.
- Direct dataset download is currently inconvenient because the original YouTube links may be unavailable/corrupted.

Next:
Do not spend much time downloading the full dataset yet. Keep it as a possible source of audio/pretraining data.


2. AVSBench S4
--------------
Status: Obtained from professor
Priority: VERY HIGH

- 4,932 real videos.
- Single sounding object/source per video.
- Synchronized audio and video.
- Pixel-level masks of the sounding object.
- This is currently the best dataset for our first experiment.

Possible use:

    Original RGB + audio + mask
              |
              v
    Mask the sounding object
              |
              v
    Partial RGB + Mel spectrogram
              |
              v
          Diffusion
              |
              v
      Reconstructed RGB

Next:
Explore the processed data received from the professor and understand exactly what files, frames, audio, masks and splits are available.


3. AVSBench MS3
----------------
Status: Obtained from professor
Priority: High

- 424 videos.
- Multiple sounding objects can be present.
- Pixel-level sounding-object masks.
- Much smaller than S4.
- Useful as a harder multi-object extension after S4 works.

Next:
Explore after understanding S4.


4. MS COCO + Captions
---------------------
Status: Unexplored
Priority: Low-Medium

- Large image dataset with object masks.
- Has captions describing the images.
- Does not contain real audio.
- Could be combined with VGGSound or generated audio to create synthetic image-audio pairs.

Main limitation:
The audio would not actually come from the particular scene/image.

Next:
Keep as an optional source for additional visual training data.


5. AVS-Synthetic
----------------
Status: Found and understood
Priority: High

Link:
https://jinxiang-liu.github.io/anno-free-AVS/

- 62,609 image-audio-mask triplets.
- 46 categories.
- 52,609 train / 5,000 validation / 5,000 test.
- Image/mask comes from existing image datasets.
- Audio comes from VGGSound.
- Image and audio are matched semantically, not recorded from the same scene.

Example:

    Dog image + dog mask + dog barking audio

Main use:
Good candidate for large-scale pretraining before fine-tuning on real S4 data.

Next:
Investigate the dataset structure and decide whether to use it for pretraining.






# ------------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------------------ #

7. MRSAudio
-----------
Status: Very promising, unexplored
Priority: VERY HIGH

- Large spatial audio-visual dataset.
- Contains egocentric and exocentric video.
- Contains binaural and FOA spatial audio.
- Also includes 3D trajectories and other annotations.
- Much closer to the original "missing view + audio" project idea.

Potential formulation:

    View A + audio
          |
          v
        Model
          |
          v
        View B

Main issue:
The project currently needs Mel-spectrogram-based audio representations, so we need to check how useful the spatial audio remains after converting it to Mel spectrograms.

Next:
Download and inspect only a very small sample first. Verify which video is ego/exo and how everything is synchronized.


8. STARSS23
-----------
Status: Very promising, unexplored
Priority: High

- Real-world spatial audio scenes.
- 360-degree video.
- Synchronized spatial audio.
- Provides FOA/MIC recordings.
- Has spatial annotations such as azimuth, elevation and distance.

Potential use:

    360-degree scene
          |
       split into
       /             View A      View B
       \         /
        \       /
         Audio
           |
           v
     Missing-view model

Next:
Inspect one scene and check whether two perspective views plus Mel audio can be extracted cleanly.


9. 360+x
--------
Status: Promising additional candidate
Priority: High

- 2,152 videos.
- About 67.78 hours.
- Contains 360-degree video.
- Contains egocentric and other camera views.
- Contains multi-channel spatial audio.

Potentially one of the strongest candidates for:

    View A + audio -> View B

Next:
Investigate after MRSAudio and STARSS23.


10. REC-Street (Haven't looked into this at all)
-------------
Status: Additional candidate
Priority: Medium-Low

- 360-degree video.
- Spatial/FOA audio.
- Useful for multi-view + spatial-audio experiments.

Next:
Keep as a backup. Investigate only if MRSAudio/360+x/STARSS23 are unsuitable.


11. AVSBench-Semantic / AVSS
----------------------------
Status: Known, not yet explored
Priority: Medium

An extension of the AVSBench family with semantic/multi-source annotations.

AVSBench family:

    S4
      Single sounding source

    MS3
      Multiple sounding sources

    AVSS / Semantic
      Multiple sources + semantic masks

Next:
S4 first, then MS3, then AVSS.


Current Priority
================

NOW
---

1. AVSBench S4
   -> Build the first working prototype.

2. AVS-Synthetic
   -> Investigate for large-scale pretraining.

3. AVSBench MS3
   -> Use later as a multi-object extension.

4. MRSAudio
   -> Investigate seriously for the original missing-view formulation.

5. 360+x
   -> Investigate seriously as another missing-view candidate.

6. STARSS23
   -> Investigate for 360-degree + spatial-audio experiments.

7. MS COCO + Captions
   -> Keep as optional synthetic/visual data.

8. AVE
   -> Optional only.


Research Direction
==================

A. Practical first version
--------------------------

Acoustic-guided visual inpainting.

    Partial RGB + Mel spectrogram
                  |
                  v
             Diffusion
                  |
                  v
       Reconstructed missing object

Best datasets:
AVS-Synthetic -> S4 -> MS3


B. Original / stronger version
------------------------------

Acoustic-guided missing-view generation.

    View A + audio
          |
          v
        Model
          |
          v
        View B

Best candidates:
MRSAudio / 360+x / STARSS23


Immediate Next Step
===================

Do not download more datasets yet.

First, understand the processed S4 data received from the professor:

- What RGB frames are available?
- What audio files are available?
- What format are the masks in?
- How are audio and frames synchronized?
- What are the train/validation/test splits?
- What preprocessing has already been done?
- What additional processed data is available?

Then build the first simple benchmark:

    S4 sample
       |
       +-- Original RGB
       +-- Sounding-object mask
       +-- Synchronized audio
                    |
                    v
              Mel spectrogram
                    |
                    v
          Mask sounding object
                    |
                    v
             Partial RGB + Mel
                    |
                    v
              Diffusion model
                    |
                    v
             Reconstructed RGB
                    |
                    v
             Original RGB
                (ground truth)

The goal of the first experiment is simply to establish whether
acoustic information helps reconstruct a visually missing object.

