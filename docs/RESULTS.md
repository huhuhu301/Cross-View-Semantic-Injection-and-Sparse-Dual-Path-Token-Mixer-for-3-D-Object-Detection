# RV-SDTM: manuscript-reported results

This page follows the supplied manuscript's displayed values and decimal
precision. Original measured records are preserved separately in
[Archived checkpoint evaluations](ARCHIVED_RESULTS.md) and
[archived_results.json](archived_results.json).

## Summary

| Dataset | Manuscript protocol | Result (%) | Source table |
|---|---|---|---|
| Waymo Open Dataset | Validation, L1 / L2 | L1 mAP/mAPH **82.9/80.6**; L2 mAP/mAPH **76.8/74.7** | `tab:waymo_sota` |
| nuScenes | LiDAR-only validation | NDS **71.9**; mAP **68.6** | `tab:nuscenes_sota` |

## Argoverse 2 implementation and evaluation

The release includes the AV2 dataset adapter, preprocessing utilities,
RV-SDTM configuration, training entry point, and official-evaluator integration
for 26 categories under the 200 m ROI-only setting.

- [AV2 model configuration](../cfgs/argoverse_models/rv_sdtm.yaml)
- [Dataset preparation](GETTING_STARTED.md#argoverse-2)
- [Training](GETTING_STARTED.md#training) and [evaluation](GETTING_STARTED.md#evaluation)
- [Archived checkpoint metrics](ARCHIVED_RESULTS.md#summary) and
  [distance diagnostics](ARCHIVED_RESULTS.md#av2-distance-diagnostics)

## Source and formatting

Source: the author-supplied `RV_SDTM_Main_Manuscript_modification_V1.zip`,
containing `RV_SDTM_Main_Manuscript.tex`. Relevant source locations are
`tab:waymo_sota` (lines 740–770) and `tab:nuscenes_sota` (772–799).
All metrics displayed in the tables above use **one decimal place**, as in
those manuscript tables. Timings, parameter counts, configuration values,
and archived records retain their original precision. Checkpoint identities
and AV2 error terms are documented in the archive; qualitative-video provenance
is documented in the gallery.

[Back to the project](../README.md) · [Detection gallery](SHOWCASE.md) ·
[Archived checkpoint evaluations](ARCHIVED_RESULTS.md)
