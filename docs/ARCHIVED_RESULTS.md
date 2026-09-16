# RV-SDTM: archived checkpoint evaluations

This page records the archived RV-SDTM checkpoint evaluations associated with
this source release. Checkpoints and raw evaluation outputs are archived
separately and are not distributed in the repository.

The same summary is available in machine-readable form as
[`archived_results.json`](archived_results.json).

## Summary

| Dataset | Checkpoint | Evaluation protocol | Result |
|---|---|---|---|
| Waymo Open Dataset | epoch 12, iteration 59,208 | official validation | L1 mAP/mAPH 82.7917/80.5876; L2 mAP/mAPH **76.8013/74.6541** |
| nuScenes | R2 epoch 24 | paired-seed-v2, seed `666 + global rank` | NDS **71.0552**; mAP **67.4185** |
| Argoverse 2 | epoch 12, iteration 82,560 | official evaluator, all 23,547 validation frames, 26 categories, 200 m ROI-only | AP **0.380**; CDS **0.295**; ATE **0.429**; ASE **0.325**; AOE **0.705** |

Additional archived nuScenes diagnostics, retained from the release guide:
Truck AP 59.0167, Trailer AP 44.3476; mATE 0.26794, mASE 0.25111,
mAOE 0.26801, mAVE 0.28996, mAAE 0.18839. AP is in percent; error terms
retain their native units.

The AV2 AP/CDS values are 38.0/29.5 when expressed as percentages. The 200 m
ROI-only setting is the paper protocol and must not be compared as though it
were the default 150 m AV2 leaderboard protocol.

The supplied manuscript draft quotes different values for some metrics.
Those values are displayed separately in [Manuscript results](RESULTS.md).
This archive preserves the measured checkpoint records and their original
precision; it is not overwritten to match the manuscript tables.

## AV2 distance diagnostics

| Radial annulus | mAP | mCDS |
|---|---:|---:|
| `[0, 50)` m | 0.525 | 0.422 |
| `[50, 100)` m | 0.255 | 0.190 |
| `[100, 150)` m | 0.107 | 0.074 |
| `[150, 200]` m | 0.036 | 0.024 |

These are custom radial-annulus diagnostics computed with the official metric
formula. Overall and annular evaluations each use their respective
ground-truth distribution.

## Checkpoint identities

- Waymo epoch 12 SHA-256:
  `f67a27ef0def225f1ce8b1348daf92b5e696b410170a0a7535e91def4d6277fc`
- nuScenes R2 epoch 24 SHA-256:
  `66132058b6e739f1a9654829998a611628d74bfcbbd110616431a14a613a10bd`
- Argoverse 2 epoch 12 SHA-256:
  `4252c4b21a9e6f23c52aaa6951b5a6d23a140c8b23433716627b0864ba47aabb`

## AV2 comparison boundary

The reference baseline evaluated in the same campaign produced AP 0.391 and CDS
0.303. That baseline is useful context only: its configuration and checkpoint
are not part of this RV-SDTM-only release.
