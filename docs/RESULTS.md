# Range–Voxel 3D Detection: validation results

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

The AV2 AP/CDS values are 38.0/29.5 when expressed as percentages. The 200 m
ROI-only setting is the paper protocol and must not be compared as though it
were the default 150 m AV2 leaderboard protocol.

The supplied manuscript draft quotes 71.9 NDS / 68.6 mAP for nuScenes and
40.4 mAP for AV2. Those numbers are not the archived checkpoint evaluations
documented here; this release uses the measured values in the table above.

## AV2 distance diagnostics

| Radial annulus | mAP | mCDS |
|---|---:|---:|
| `[0, 50)` m | 0.525 | 0.422 |
| `[50, 100)` m | 0.255 | 0.190 |
| `[100, 150)` m | 0.107 | 0.074 |
| `[150, 200]` m | 0.036 | 0.024 |

These are custom radial-annulus diagnostics computed with the official metric
formula. They are not independent AV2 leaderboard submissions, and their
average is not the overall metric because each annulus has a different
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

The paper-constrained 40.2/40.4 AV2 table and its projected per-distance or
per-class values were synthetic planning artifacts. They are neither evaluator
outputs nor empirical measurements and are intentionally absent from this
release. Do not cite them as experimental results.
