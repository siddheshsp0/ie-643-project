# Data

The datasets are not in the repository (about 37 GB, and AVSBench is CC-BY-NC-4.0). Put them here:

```
data/
  avsbench_data_processed/Single-source/
    s4_meta_data.csv                       name, start, category, split
    s4_data/audio_wav/{train,val,test}/<category>/<name>.wav
    s4_data/visual_frames/{train,val,test}/<category>/<name>/<name>_{1..5}.png
    s4_data/gt_masks/{train,val,test}/<category>/<name>/<name>_{k}.png
  cache/clap_s4/{train,val,test}.pt        created by scripts/cache_clap_features.py (699 MB)
  scene-classification-images-and-audio/images/   optional, only for outputs/samples/
```

- **AVSBench single-source (S4)**: 4,932 five-second YouTube clips in 23 classes, with five 224 × 224
  frames per clip, the clip's audio and sounding-object masks (train: frame 1 only; val and test: all 5).
  Official splits: 3,452 / 740 / 740. Project page: https://opennlplab.github.io/AVSBench/. We used the
  processed S4 release (`s4_data`, `s4_meta_data.csv`), obtained through correspondence with the
  AVSBench authors.
- **Scene classification: images and audio** (Kaggle, Jordan J. Bird):
  https://www.kaggle.com/datasets/birdy654/scene-classification-images-and-audio. We only used five photos
  from it (restaurant3-43, store3-160, village1-154, forest662, classroom5-512) as unseen real scenes.

After placing S4, run `python scripts/cache_clap_features.py` once (about 2 minutes on a GPU). Training
and `scripts/eval_inpainting.py` read the cached features.
