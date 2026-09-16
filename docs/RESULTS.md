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
| Argoverse 2† | 200 m ROI-only, 26-category macro average | mAP/CDS **40.4/31.4** | `tab:av2_sota`, `tab:av2_distance` |

† AV2 entries are **synthetic target projections** for manuscript comparison.
Measured checkpoint results are recorded separately in the archive.

## AV2 distance table · manuscript target projections

Source: `tab:av2_distance`. Values are mAP/CDS in percent and preserve the
manuscript's one-decimal formatting, including trailing zeros. The table
compares the two methods' manuscript target projections across five ranges.

| Cuboid-center distance | FSHNet target mAP/CDS (%) | RV-SDTM target mAP/CDS (%) |
|---|---:|---:|
| Overall `[0, 200]` m | 40.2/31.2 | **40.4/31.4** |
| `[0, 50)` m | 55.2/44.4 | **55.3/44.5** |
| `[50, 100)` m | 27.8/20.6 | **28.0/20.9** |
| `[100, 150)` m | 12.1/8.4 | **12.4/8.5** |
| `[150, 200]` m | 4.2/2.8 | **4.6/3.1** |

The manuscript specifies a fixed 26-category macro average in every interval
under its 200 m ROI-only setting. Measured checkpoint distance diagnostics
remain in the
[archive](ARCHIVED_RESULTS.md#av2-distance-diagnostics).

## Source and formatting

Source: the author-supplied `RV_SDTM_Main_Manuscript_modification_V1.zip`,
containing `RV_SDTM_Main_Manuscript.tex`. Relevant source locations are
`tab:waymo_sota` (lines 740–770), `tab:nuscenes_sota` (772–799),
`tab:av2_sota` (872–986), and `tab:av2_distance` (1259–1283).
All metrics displayed in the tables above use **one decimal place**, as in
those manuscript tables. Timings, parameter counts, configuration values,
and archived records retain their original precision. Checkpoint identities
and AV2 error terms are documented in the archive; qualitative-video provenance
is documented in the gallery.

[Back to the project](../README.md) · [Detection gallery](SHOWCASE.md) ·
[Archived checkpoint evaluations](ARCHIVED_RESULTS.md)
