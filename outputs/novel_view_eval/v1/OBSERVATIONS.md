# Observations (hand-written, from the contact sheets in `sheets/`)

The numbers are in `README.md`, which `scripts/eval_novel_view.py report` regenerates. This file is not regenerated.

1. **The sound decides what appears in large revealed regions.** At +15° the camera moves right and turns left. The band that rotates into view and the space behind near objects get the sound's object (cat, dog, baby, guitar), while the no-audio baseline paints people or background. Best: `bathroom_door` (92% CLIP top-1 hits) and `country_path` (83%). The paired win rate is 77–90% for every sound.
2. **−15° is weaker (38–62% vs 58–79% at +15°).** MegaScenes' clockwise orbit also moves the camera *forward* (`tz = sin θ < 0`). The view zooms in and blurs (S4 frames are 224 px), and the unseen area shrinks to slivers around near objects (e.g. 12% unseen on `bus_street`).
3. **Thin holes hugging a foreground object are completed as that object**, with or without audio: dog-shaped gaps extend the dog in `doorway_dog`, gaps beside the violinist extend her in `piano_room`, and gaps around the bus extend the bus at −15° in `bus_street`. The sound only wins when the hole has room for a new object. Still, `doorway_dog` +15° shows a cat tail, a guitar and a baby partly behind the dog for some seeds.
4. **Text/watermark artifacts** (captions, URLs, logos) appear in many images, noticeably more often with audio than without in the sheets inspected. A likely cause is that the adapter was trained on S4 YouTube frames, many of which carry overlays.
5. **Confusions:** cat audio sometimes yields a dog (cat → P(dog) = 0.15 on average); the other sounds rarely cross over.
6. **CLIP-metric caveats:** the score sees only the generated pixels. It undercounts objects held by a generated person: guitars in `piano_room` are labelled "woman singing". It also occasionally scores a dark blob as an animal. Treat it as a coarse, conservative check and use the sheets for judgement.
7. **Hole rule** (`sheets/hole_rules.png`): MegaScenes' angle rule drops 47–57% of each +15° view, shredding walls, floors and foliage. The stretch rule used here drops 30–44%, keeping every surface the input saw.
