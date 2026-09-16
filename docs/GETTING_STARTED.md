# Getting started

This repository provides one RV-SDTM configuration for each supported
dataset. This guide covers data preparation and the commands needed to train
and evaluate a model. For checkpoint identities and compatibility details, see
the [release guide](RV_SDTM_RELEASE.md).

## Canonical configurations

Use these paths from the repository root:

| Dataset | Configuration |
|---|---|
| Waymo Open Dataset | `cfgs/waymo_models/rv_sdtm.yaml` |
| nuScenes | `cfgs/nuscenes_models/rv_sdtm.yaml` |
| Argoverse 2 | `cfgs/argoverse_models/rv_sdtm.yaml` |

The matching files under `tools/cfgs/` are for commands launched from the
`tools/` directory. Always pass an explicit configuration path.

## Installation

The recommended stack is Ubuntu 18.04, Python 3.8.20, PyTorch 1.10.0+cu113,
and CUDA 11.3, matching the validated local A10 server. See
[INSTALL.md](INSTALL.md), then build the CUDA extensions:

```bash
pip install -r requirements.txt
python setup.py develop
```

Dataset downloads are not redistributed by this repository. Follow the
license and download instructions of each dataset owner.

## Waymo Open Dataset

Place TFRecords under `data/waymo/raw_data/`, then create processed sequences,
infos, and the GT database:

```bash
python -m pcdet.datasets.waymo.waymo_dataset \
    --func create_waymo_infos \
    --cfg_file cfgs/dataset_configs/waymo_dataset.yaml \
    --data_path data/waymo --save_path data/waymo \
    --processed_data_tag waymo_processed_data_v0_5_0 \
    --workers 16
```

The data and save roots must be identical so database-sampler paths remain
relative to the dataset root. Install official Waymo evaluation dependencies
with `pip install -r requirements-waymo.txt` when needed.

## nuScenes

Place the official train/validation release under
`data/nuscenes/v1.0-trainval/`, then run:

```bash
python -m pcdet.datasets.nuscenes.nuscenes_dataset \
    --func create_all \
    --cfg_file cfgs/nuscenes_models/rv_sdtm.yaml \
    --data_path data/nuscenes --save_path data/nuscenes \
    --version v1.0-trainval --max_sweeps 10
```

This creates ten-sweep infos, per-object GT files, DB infos, and the global
point backing array. The data and save roots must be identical when creating
the GT database.

## Argoverse 2

Place the raw Sensor Dataset at `data/argo2/sensor/train/` and
`data/argo2/sensor/val/`, then run:

```bash
python -m pcdet.datasets.argo2.argo2_dataset \
    --func create_all \
    --root_path data/argo2 \
    --cfg_file cfgs/argoverse_models/rv_sdtm.yaml
```

This creates deterministic train/validation infos, `ImageSets`, converted
point files, `val_anno.feather`, per-object GT files, DB infos, and the global
point backing array.

## Training

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

Replace the model configuration with the nuScenes or AV2 canonical path for
those datasets. YAML supplies the per-GPU batch size unless a global
`--batch_size` is provided.

The archived AV2 run used eight A10 GPUs, global batch size 16, seed 666,
and `RV_VOX_CHUNK=512`. Epoch 1 used AMP; the resumed epochs 2–12 used FP32.
The reference command below uses FP32 throughout:

```bash
(cd tools && bash scripts/dist_train.sh 8 \
    --cfg_file cfgs/argoverse_models/rv_sdtm.yaml \
    --batch_size 16 --epochs 12 --workers 1 \
    --fix_random_seed --skip_post_eval \
    --extra_tag av2_e12_seed666 \
    --set MODEL.BACKBONE_3D.RV_VOX_CHUNK 512)
```

Omitting `--use_amp` selects FP32. The archived job resumed complete training
state from epoch 1 before continuing to epoch 12; a fresh run from random
initialization is not claimed to be bitwise identical.

## Evaluation

The released router uses stochastic token routing at evaluation time. Use a
fixed seed and identical data ordering for paired comparisons:

```bash
python tools/test.py \
    --cfg_file cfgs/nuscenes_models/rv_sdtm.yaml \
    --ckpt /path/to/checkpoint_epoch_24.pth \
    --fix_random_seed --eval_seed 666 \
    --root_dir . --output_dir output/nuscenes_rv_sdtm_eval
```

Archived checkpoints can be used for evaluation and model-only initialization.
The archived Waymo and nuScenes optimizer states are incompatible with the
released optimizer grouping; use `--pretrained_model` when initializing
training from those checkpoints. The archived AV2 checkpoint and checkpoints
created by this release support full resume with `--ckpt`. See the
[compatibility guide](RV_SDTM_RELEASE.md#4-historical-checkpoint-compatibility).

For the archived AV2 evaluation settings, run on four GPUs with global batch
size 8, seed 666, and the same runtime chunk override:

```bash
(cd tools && bash scripts/dist_test.sh 4 \
    --cfg_file cfgs/argoverse_models/rv_sdtm.yaml \
    --ckpt /path/to/checkpoint_epoch_12.pth \
    --batch_size 8 --workers 1 \
    --fix_random_seed --eval_seed 666 \
    --root_dir .. --output_dir ../output/av2_rv_sdtm_eval \
    --set MODEL.BACKBONE_3D.RV_VOX_CHUNK 512)
```

This evaluates the configured 200 m ROI-only paper protocol. It is not the
default 150 m AV2 leaderboard protocol. The canonical config makes this
explicit with `DATA_CONFIG.EVALUATE_RANGE: 200.0` and
`DATA_CONFIG.EVAL_ONLY_ROI_INSTANCES: True`. See [RESULTS.md](RESULTS.md) for
the measured metrics and checkpoint SHA-256.

## Dataset-free verification

```bash
python tools/validate_rv_sdtm_release.py
python tools/validate_rv_sdtm_release.py --build
python tools/validate_rv_sdtm_release.py --argo2-forward
python -m compileall pcdet tools
```

The last two validation modes require a CUDA device. They do not read dataset
files or reproduce an official metric. A fresh-machine end-to-end run remains
separate from these dataset-free checks.
