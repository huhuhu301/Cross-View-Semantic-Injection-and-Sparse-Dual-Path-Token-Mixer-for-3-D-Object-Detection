# RV-SDTM: results

RV-SDTM supports LiDAR-based 3D object detection on Waymo, nuScenes, and
Argoverse 2. The table combines their evaluation settings and summary figures.

## Summary

| Dataset | Evaluation setting | Result (%) |
|---|---|---|
| Waymo Open Dataset | Validation, L1 / L2 | L1 mAP/mAPH **82.9/80.6**; L2 mAP/mAPH **76.8/74.7** |
| nuScenes | LiDAR-only validation | NDS **71.9**; mAP **68.6** |
| Argoverse 2 | Validation, 26 classes, 200 m, ROI-only | mAP **40.4** |

## Reproduction resources

- Configurations: [Waymo](../cfgs/waymo_models/rv_sdtm.yaml),
  [nuScenes](../cfgs/nuscenes_models/rv_sdtm.yaml),
  [Argoverse 2](../cfgs/argoverse_models/rv_sdtm.yaml)
- [Dataset preparation](GETTING_STARTED.md)
- [Training](GETTING_STARTED.md#training) and [evaluation](GETTING_STARTED.md#evaluation)

[Back to the project](../README.md) · [Detection gallery](SHOWCASE.md)
