# RV-SDTM: detection gallery

Selected Waymo visualizations: one GT-coverage comparison, a full-scene BEV
sequence, and four multi-view prediction videos. No model execution is needed
to view these author-supplied materials.

[Coverage](#gt-coverage-in-dense-traffic) · [BEV](#full-scene-bev) ·
[Multi-view videos](#multi-view-videos) · [Viewing options](#viewing-options) ·
[Provenance](#provenance-and-diagnostic-protocol) · [Measured results](RESULTS.md)

## GT coverage in dense traffic

![GT coverage: RV-SDTM-only matches in green, baseline-only in red, both in light gray, neither in dark gray.](showcase/assets/b_coverage.png)

[Full-size image](showcase/assets/b_coverage.png)

The image compares **GT matching status**, not predicted box coordinates.
Its “FSHNet” display label means **FSHNet-Light**, not FSHNet-Base.

| Color | Meaning |
|---|---|
| Green | GT matched only by RV-SDTM |
| Red | GT matched only by FSHNet-Light |
| Light gray | GT matched by both models |
| Dark gray | GT matched by neither model |
| Gold outline | Area enlarged in the right-hand panel |

All valid GT in the displayed area remain visible. Unmatched prediction
markers are hidden, so this image cannot establish an absence of false
positives. The displayed frame was selected for the largest net GT-coverage
gain among 30 candidate frames, with earlier frames breaking ties; it is not
an unbiased validation-set sample.

## Full-scene BEV

[![Animated BEV preview with blue RV-SDTM predicted boxes.](showcase/assets/full_bev.gif)](showcase/assets/full_bev.mp4)

[Open / download 1080p MP4 · 12 seconds](showcase/assets/full_bev.mp4)

Blue boxes are actual RV-SDTM predictions at a display score threshold of 0.5.
The sequence concatenates four different scenes, not one continuous drive.
Each source clip contains 30 consecutive frames at 10 Hz; each frame is shown
three times in the 30 fps video. This is playback timing, **not inference FPS**.

## Multi-view videos

| Cyclists | Pedestrians |
|:---:|:---:|
| [![Cyclists video cover.](showcase/assets/01_cyclists.jpg)](showcase/assets/01_cyclists.mp4) | [![Pedestrians video cover.](showcase/assets/02_pedestrians.jpg)](showcase/assets/02_pedestrians.mp4) |
| [1080p MP4 · 13 s](showcase/assets/01_cyclists.mp4) | [1080p MP4 · 13 s](showcase/assets/02_pedestrians.mp4) |
| **Distant pedestrians & cyclists** | **Dense traffic** |
| [![Distant pedestrians and cyclists video cover.](showcase/assets/03_far_vru.jpg)](showcase/assets/03_far_vru.mp4) | [![Dense traffic video cover.](showcase/assets/04_vehicles.jpg)](showcase/assets/04_vehicles.mp4) |
| [1080p MP4 · 13 s](showcase/assets/03_far_vru.mp4) | [1080p MP4 · 13 s](showcase/assets/04_vehicles.mp4) |

Each video replays one source clip from an elevated oblique view, a low-angle
view, a **FROZEN-FRAME ORBIT**, and BEV. The orbit moves the camera around a
frozen detection frame; it does not add new detection timesteps. These videos
reuse the same four scenes as the BEV sequence, not four additional scenes.
No GT-based prediction removal, extra NMS, tracking, target interpolation, or
generated boxes were applied to the supplied videos.

## Viewing options

On GitHub, the GIF and cover images are visible directly in Markdown. Click a
cover or MP4 link to open the asset; if a player is unavailable, download the
original file. Repository access is required while this project is private.

A self-contained, responsive HTML gallery with five video players is included
in [showcase/](showcase/README.md). After cloning, run from the repository root:

```bash
python3 -m http.server 8770 --bind 127.0.0.1 --directory docs/showcase
```

Open `http://127.0.0.1:8770/` on that machine. Alternatively, open the downloaded
`docs/showcase/index.html` in a browser. GitHub's HTML file view shows source,
not the rendered gallery. **No public Pages site is enabled by this update.**
The gallery has no CDN, analytics, model, or dataset dependency.

## Provenance and diagnostic protocol

The source handoff records these model identities; weights are not included:

| Role | Source configuration | Checkpoint SHA-256 prefix | Loaded tensors reported by the source |
|---|---|---|---:|
| RV-SDTM visualizations | `fshnet_sdtm_best.yaml` | `248718635f90…` | 902 / 902 |
| Comparison baseline | `fshnet_light.yaml` | `168fec20bd59…` | 377 / 377 |

Full original configuration/checkpoint names, hashes, source-frame identities,
frame mappings, and unchanged media hashes are in [metadata.json](showcase/metadata.json).
Historical filenames document provenance; they are not release model entrypoints.
**The visualization checkpoint differs from the archived Waymo benchmark
checkpoint** (`f67a27ef0def…`) in [Results](RESULTS.md). The handoff does not
provide official benchmark scores for its visualization checkpoint.

For the coverage diagnostic, both models use score threshold **0.5**, their
own training data configurations and native postprocessing, with aligned source
frames. Same-class, one-to-one 3D IoU matching uses thresholds **0.7 / 0.5 / 0.5**
for Vehicle / Pedestrian / Cyclist. GT must contain at least one LiDAR point.
The display region is the XY square **|x|, |y| ≤ 75 m**, not a radius-75 m circle.
An IoU failure can also produce an unmatched GT; coverage is not simply a
classification-miss count and does not implement the complete official AP/APH protocol.

The selected vehicle-scene frame is **0146**. It has 36 RV-SDTM-only matches,
8 baseline-only matches, 70 shared matches, and 32 GT matched by neither
model. The net difference is +28 for this selected frame, not an aggregate
benchmark gain. Unmatched prediction counts remain in the
[complete 30-frame diagnostic CSV](showcase/b_coverage_scores.csv).

All 12 included media files are copied byte-for-byte from the supplied package.
Four redundant multi-view GIFs are omitted; the original MP4s and posters are
retained. No architecture figures, raw point clouds, annotations, checkpoints,
or delivery archive are included. Dataset/media rights are not relicensed by
this repository; public distribution remains subject to the applicable permissions.

## Earlier qualitative figures

These earlier author-provided manuscript figures are retained separately.
**Upper row:** FSHNet comparison. **Lower row:** RV-SDTM. Here, unlike the
coverage figure above, green denotes **predictions**, red denotes **GT**, and
blue dashed outlines highlight selected objects. The new video package's
model identities are not assigned to these earlier figures.

### Long-range sparse returns

![Earlier selected Waymo long-range scenes: comparison above, RV-SDTM below.](figures/waymo_long_range.png)

### Partial occlusion

![Earlier selected Waymo occlusion scenes: comparison above, RV-SDTM below.](figures/waymo_occlusion.png)

These selected figures are not regenerated by the release training/evaluation
commands. See [Results](RESULTS.md) for aggregate measurements and protocols.

[Back to the project](../README.md).
