# RV-SDTM media gallery

This directory contains a self-contained HTML gallery: one GT-coverage still,
one BEV video, and four multi-view videos. See the rendered GitHub
[gallery guide](../SHOWCASE.md) for previews, viewing options, and provenance.

To view the HTML page locally, run from the repository root:

```bash
python3 -m http.server 8770 --bind 127.0.0.1 --directory docs/showcase
```

Open `http://127.0.0.1:8770/` on the same machine for browser playback.
GitHub provides the Markdown gallery and downloadable media in the
repository; the HTML gallery is ready for local viewing.

The original five MP4s, six covers/stills, and one BEV GIF are retained without
re-encoding. Four redundant multi-view GIFs from the handoff are not vendored;
`metadata.json` lists the 12 retained assets and preserves source provenance.
`b_coverage_scores.csv` retains the complete original 30-frame diagnostic.

The baseline display alias “FSHNet” means FSHNet-Light. The coverage rectangles
come from GT; video blue boxes are predictions. The multi-view clips highlight
Cyclist, Pedestrian, and Vehicle detections. See [protocol.html](protocol.html)
or the [gallery guide](../SHOWCASE.md)
for full interpretation, selection rules, and checkpoint distinctions.

Media and dataset assets remain subject to their original permissions.
Confirm applicable dataset and institutional permissions before public hosting.
