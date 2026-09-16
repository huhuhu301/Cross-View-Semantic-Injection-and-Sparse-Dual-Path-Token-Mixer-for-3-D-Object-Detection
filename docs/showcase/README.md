# RV-SDTM media gallery

This directory contains a self-contained HTML gallery: one GT-coverage still,
one BEV video, and four multi-view videos. See the rendered GitHub
[gallery guide](../SHOWCASE.md) for previews, viewing options, and provenance.

To view the HTML page locally, run from the repository root:

```bash
python3 -m http.server 8770 --bind 127.0.0.1 --directory docs/showcase
```

Open `http://127.0.0.1:8770/` on the same machine. No build or model is needed.
GitHub's file browser does not render `index.html` as a website. No Pages
deployment is configured or implied here; repository visibility is unchanged.

The original five MP4s, six covers/stills, and one BEV GIF are retained without
re-encoding. Four redundant multi-view GIFs from the handoff are not vendored;
`metadata.json` lists the 12 retained assets and preserves source provenance.
`b_coverage_scores.csv` retains the complete original 30-frame diagnostic.

The baseline display alias “FSHNet” means FSHNet-Light. The coverage rectangles
come from GT; video blue boxes are predictions. Playback FPS is not inference
FPS. See [protocol.html](protocol.html) or the [gallery guide](../SHOWCASE.md)
for full interpretation, selection rules, and checkpoint distinctions.

Media/dataset rights are not granted or relicensed by the source-code license.
Confirm applicable dataset and institutional permissions before public hosting.
