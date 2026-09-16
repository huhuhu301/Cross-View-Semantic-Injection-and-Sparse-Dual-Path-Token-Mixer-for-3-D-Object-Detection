# RV-SDTM: release guide

This repository accompanies *Cross-View Semantic Injection and Sparse Dual-Path
Token Mixer for 3-D Object Detection* (RV-SDTM). This guide records the released
protocols, archived results, and checkpoint compatibility.

## 1. Supported protocols

| Dataset | Detector | Backbone | Head | Epochs | Status |
|---|---|---|---|---:|---|
| Waymo | CenterPoint | `RVSDTM` | `SparseDynamicHead` | 12 | checkpoint-compatible evaluation and corrected retraining config |
| nuScenes | TransFusion | `RVSDTMNuScenes` | `TransFusionHead` | 24 | checkpoint-compatible evaluation and corrected retraining config |
| Argoverse 2 | CenterPoint | `RVSDTMAV2` | `SparseDynamicHead` | 12 | archived official-evaluator validation and portable retraining config |

The three paths share RangeViewVFE, early RV-to-voxel GeoCVA fusion, SDTM,
attentional pillar pooling, and AFD. Dataset adapters preserve the required BEV
geometry: Waymo and nuScenes use their archived layouts; AV2 produces stride-4,
64-channel sparse BEV features for a 400 m by 400 m range.

Canonical configs when running from the repository root:

- `cfgs/waymo_models/rv_sdtm.yaml`
- `cfgs/nuscenes_models/rv_sdtm.yaml`
- `cfgs/argoverse_models/rv_sdtm.yaml`

Equivalent canonical configs beneath `tools/cfgs/` use paths relative to the
`tools/` working directory.

## 2. Release scope and provenance

This directory is a clean source-code snapshot initialized with a new Git
history. It excludes the internal working tree and its historical commits,
datasets, checkpoints, generated runs, logs, and prebuilt architecture-specific
binaries. Display tables and dataset-specific implementation links are
documented in [Results](RESULTS.md); measured checkpoint records are
preserved in the evaluation archive.

The public YAML files are portable defaults, not byte-identical exports of
machine-local launch configurations. In particular:

- the archived Waymo checkpoint uses an RV feature shape of 64 by 1024, while
  the manuscript describes 64 by 720;
- the measured AV2 run required `RV_VOX_CHUNK=512`; the canonical public YAML
  records 512, and the commands below repeat it explicitly so local overrides
  cannot silently change that memory/protocol setting;
- shared GT-database memory is disabled by default so the configs do not depend
  on host-local shared-memory artifacts;
- no private filesystem path, host name, or distributed master address is part
  of the public protocol.

The source has dataset-free configuration, build, and synthetic forward/backward
checks in the recommended local A10 environment, but a fresh-machine
end-to-end conversion, train, and official-evaluation cycle is not claimed.

## 3. Archived evidence

The original measured values and precision are centralized in
[Archived checkpoint evaluations](ARCHIVED_RESULTS.md) and
[archived_results.json](archived_results.json). The
[manuscript display tables](RESULTS.md) are not substituted for this evidence
or assigned to these checkpoint hashes. This section records run identities
and compatibility details only.

### Waymo

- checkpoint: epoch 12, iteration 59,208
- checkpoint SHA-256:
  `f67a27ef0def225f1ce8b1348daf92b5e696b410170a0a7535e91def4d6277fc`
- measured scores: [archived summary](ARCHIVED_RESULTS.md#summary)
- the archived checkpoint uses an RV feature shape of 64 by 1024, while the
  manuscript describes 64 by 720
- the native official far-distance bin is `[50, +inf)`; `[50, 75)` is a
  separate companion analysis

### nuScenes

- checkpoint: R2 epoch 24
- checkpoint SHA-256:
  `66132058b6e739f1a9654829998a611628d74bfcbbd110616431a14a613a10bd`
- official validation protocol: paired-seed-v2, seed `666 + global rank`
- measured scores and class/error diagnostics:
  [archived summary](ARCHIVED_RESULTS.md#summary)

### Argoverse 2

- checkpoint: epoch 12, iteration 82,560
- checkpoint SHA-256:
  `4252c4b21a9e6f23c52aaa6951b5a6d23a140c8b23433716627b0864ba47aabb`
- official evaluator over all 23,547 unique validation frames and all 26
  categories under the 200 m ROI-only paper protocol
- measured scores: [archived summary](ARCHIVED_RESULTS.md#summary)
- evaluation used paired seed 666, four A10 GPUs, global evaluation batch 8,
  and runtime `RV_VOX_CHUNK=512`
- training used eight A10 GPUs, global batch 16, seed 666, and runtime
  `RV_VOX_CHUNK=512`; epoch 1 used AMP, then the run resumed model and optimizer
  state and trained epochs 2–12 in FP32. The restored RNG state was not
  bitwise-continuous with epoch 1
- recorded environment: Ubuntu 18.04, Python 3.8.20, PyTorch 1.10.0+cu113,
  torchvision 0.11.0+cu113, CUDA 11.3, cuDNN 8.2, spconv-cu113 2.3.6,
  torch-scatter 2.1.2, and av2 0.2.1

This is an official-evaluator result under the paper's 200 m ROI-only setting,
not the default 150 m AV2 leaderboard protocol. The reference baseline's
measured scores remain in the [archive](ARCHIVED_RESULTS.md#av2-comparison-boundary);
its configuration and checkpoint are outside this RV-SDTM-only release.

The public protocol encodes that boundary as
`DATA_CONFIG.EVALUATE_RANGE: 200.0` and
`DATA_CONFIG.EVAL_ONLY_ROI_INSTANCES: True`; do not omit or reinterpret these
keys when comparing against the archived result.

The custom official-formula distance diagnostics remain in the
[archived distance table](ARCHIVED_RESULTS.md#av2-distance-diagnostics).
Overall and annular evaluations each use their respective ground-truth
distribution.

## 4. Historical checkpoint compatibility

The archived Waymo, nuScenes, and Argoverse 2 model state dictionaries load
with no missing, unexpected, or shape-mismatched tensors (914/914, 1092/1092,
and 1195/1195 tensors respectively). Use them as follows:

- evaluation: `tools/test.py --ckpt ...`
- model-only training initialization: `tools/train.py --pretrained_model ...`

Do not use the archived Waymo or nuScenes checkpoints with
`tools/train.py --ckpt` for a full optimizer resume. The public optimizer
builder fixes omitted direct parameters (including GeoCVA scalars and attention
projection parameters), so its parameter groups intentionally differ from
those two historical optimizer states. Use `--pretrained_model` for model-only
initialization. The archived AV2 checkpoint and new checkpoints created by this
release use the corrected optimizer grouping and can be resumed with `--ckpt`.

The canonical YAML files should therefore be understood as checkpoint-compatible
evaluation protocols plus corrected retraining configurations. A fresh run is
not claimed to be byte-for-byte identical to the historical training process.

## 5. Dataset preparation

Run these commands from the repository root. Dataset downloads and licenses are
managed by their respective owners.

### Waymo Open Dataset

Place TFRecords under `data/waymo/raw_data/`, then run:

```bash
python -m pcdet.datasets.waymo.waymo_dataset \
    --func create_waymo_infos \
    --cfg_file cfgs/dataset_configs/waymo_dataset.yaml \
    --data_path data/waymo --save_path data/waymo \
    --processed_data_tag waymo_processed_data_v0_5_0 \
    --workers 16
```

The command writes the processed sequences, train/validation info files,
per-object GT files, DB infos, and the optional global `.npy` backing array.
The CPU-parallel GT-only path is also available:

```bash
python -m pcdet.datasets.waymo.waymo_dataset \
    --func create_waymo_gt_database \
    --cfg_file cfgs/dataset_configs/waymo_dataset.yaml \
    --data_path data/waymo --save_path data/waymo \
    --processed_data_tag waymo_processed_data_v0_5_0 \
    --split train --use_parallel --workers 16
```

### nuScenes

Place the official dataset under `data/nuscenes/v1.0-trainval/`, then run:

```bash
python -m pcdet.datasets.nuscenes.nuscenes_dataset \
    --func create_all \
    --cfg_file cfgs/nuscenes_models/rv_sdtm.yaml \
    --data_path data/nuscenes --save_path data/nuscenes \
    --version v1.0-trainval --max_sweeps 10
```

The GT sampler artifacts are stored inside the version directory:

- `gt_database_10sweeps_withvelo/*.bin`
- `nuscenes_dbinfos_10sweeps_withvelo.pkl`
- `nuscenes_10sweeps_withvelo_lidar.npy`

`--data_path` and `--save_path` must be identical when creating the database so
relative sampler paths remain valid.

### Argoverse 2

Place raw sensor splits at `data/argo2/sensor/train/` and
`data/argo2/sensor/val/`, then run:

```bash
python -m pcdet.datasets.argo2.argo2_dataset \
    --func create_all \
    --root_path data/argo2 \
    --cfg_file cfgs/argoverse_models/rv_sdtm.yaml
```

This writes deterministic train/validation info files, `ImageSets`, converted
point files, `val_anno.feather`, per-object GT files, DB infos, and
`argo2_dbinfos_global.npy`.

## 6. Training

Single GPU from the repository root:

```bash
python tools/train.py \
    --cfg_file cfgs/waymo_models/rv_sdtm.yaml \
    --fix_random_seed --skip_post_eval
```

Waymo reference training on four GPUs from `tools/`:

```bash
(cd tools && bash scripts/dist_train.sh 4 \
    --cfg_file cfgs/waymo_models/rv_sdtm.yaml \
    --batch_size 8 --epochs 12 --workers 2 \
    --fix_random_seed --sync_bn --skip_post_eval \
    --extra_tag waymo_e12_seed666)
```

Use the corresponding `nuscenes_models/` or `argoverse_models/` config to train
another dataset. `BATCH_SIZE_PER_GPU` is read from YAML unless a global
`--batch_size` is supplied.

The AV2 reference recipe below preserves the archived global batch and memory
settings. `--batch_size` is global; omitting `--use_amp` uses FP32 throughout.
The archived run used AMP for epoch 1 and FP32 after its epoch-1 resume.

```bash
(cd tools && bash scripts/dist_train.sh 8 \
    --cfg_file cfgs/argoverse_models/rv_sdtm.yaml \
    --batch_size 16 --epochs 12 --workers 1 \
    --fix_random_seed --skip_post_eval \
    --extra_tag av2_e12_seed666 \
    --set MODEL.BACKBONE_3D.RV_VOX_CHUNK 512)
```

The archived job used an eight-GPU, two-node topology and resumed full
optimizer/scheduler state at epoch 1. The command above records the effective
public hyperparameters but does not promise bitwise identity for a fresh
single-node or from-scratch run.

## 7. Evaluation

The SDTM router retains `HYBRID_RANDOM_RATIO: 0.25` during evaluation. Any
official comparison must use the same seed and data ordering:

```bash
python tools/test.py \
    --cfg_file cfgs/nuscenes_models/rv_sdtm.yaml \
    --ckpt /path/to/checkpoint_epoch_24.pth \
    --fix_random_seed --eval_seed 666 \
    --root_dir . --output_dir output/nuscenes_rv_sdtm_eval
```

For the native Waymo metrics binary, optionally add both
`--waymo_metrics_binary /abs/path/compute_detection_metrics_main` and
`--waymo_gt_bin /abs/path/gt.bin`. If unavailable or invalid, the evaluator
falls back to the official Python implementation.

The archived AV2 evaluation used:

```bash
(cd tools && bash scripts/dist_test.sh 4 \
    --cfg_file cfgs/argoverse_models/rv_sdtm.yaml \
    --ckpt /path/to/checkpoint_epoch_12.pth \
    --batch_size 8 --workers 1 \
    --fix_random_seed --eval_seed 666 \
    --root_dir .. --output_dir ../output/av2_rv_sdtm_eval \
    --set MODEL.BACKBONE_3D.RV_VOX_CHUNK 512)
```

## 8. Pre-publication verification

The validation script is dataset-free by default:

```bash
python tools/validate_rv_sdtm_release.py
python tools/validate_rv_sdtm_release.py --build
python tools/validate_rv_sdtm_release.py --argo2-forward
python -m compileall pcdet tools
git diff --check
```

`--build` requires a CUDA device and validates all three model registries,
config contracts, regression layouts, class partitions, and exact optimizer
coverage. `--argo2-forward` additionally runs dataset-free synthetic inference
and a training backward pass through RV, GeoCVA, SDTM, APPool, AFD, and the
detection head. Neither command loads a checkpoint or reproduces a dataset
metric. Passing these checks is not equivalent to a fresh-machine end-to-end
reproduction.

## 9. Release contents

The source release contains the OpenPCDet substrate required by the three
protocols, the three RV-SDTM backbones/configs, preprocessing/evaluation entry
points, and this audit script. It excludes local datasets, checkpoints, output
runs, experiment scratch files, logs, and prebuilt architecture-specific CUDA
binaries. CUDA extensions must be rebuilt with `python setup.py develop`.

Licensing and provenance notices are in the root `NOTICE` and
`THIRD_PARTY_NOTICES.md` files. Manuscript display values are summarized in
[Results](RESULTS.md); unchanged measured records remain in
[Archived checkpoint evaluations](ARCHIVED_RESULTS.md).
