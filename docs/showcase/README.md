# RV-SDTM media gallery

This directory contains a self-contained HTML gallery: one GT-coverage still,
one BEV video, and four multi-view videos.
[Open the online video gallery](https://huhuhu301.github.io/Cross-View-Semantic-Injection-and-Sparse-Dual-Path-Token-Mixer-for-3-D-Object-Detection/).
See the rendered GitHub
[gallery guide](../SHOWCASE.md) for previews, viewing options, and provenance.

To view the HTML page locally, run from the repository root:

```bash
python3 -m http.server 8770 --bind 127.0.0.1 --directory docs/showcase
```

Open `http://127.0.0.1:8770/` on the same machine for browser playback.
GitHub Pages hosts the same gallery with video controls and fullscreen playback.
The README shows animated GIF previews linked to the corresponding video players.

The original five MP4s, six covers/stills, and one BEV GIF are retained without
re-encoding. Four additional 480×270 GIF previews are derived from the original
multi-view MP4s for inline README playback.
`b_coverage_scores.csv` retains the complete original 30-frame diagnostic.

The coverage rectangles
come from GT; video blue boxes are predictions. The multi-view clips highlight
Cyclist, Pedestrian, and Vehicle detections. See [protocol.html](protocol.html)
or the [gallery guide](../SHOWCASE.md)
for interpretation and scene-selection rules.

Media and dataset assets remain subject to their original permissions.
Confirm applicable dataset and institutional permissions before public hosting.
