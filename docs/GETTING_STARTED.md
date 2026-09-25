# Getting started

This guide covers dataset preparation, training and evaluation for Waymo,
nuScenes and Argoverse 2. Module definitions and Waymo ablation configurations
are described in the [implementation guide](RV_SDTM_RELEASE.md).

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

Training has been validated on NVIDIA A10 and A100, and inference/runtime
testing on NVIDIA GeForce RTX 4090. See the
[hardware validation record](INSTALL.md#hardware-validation).

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

AV2 reference training uses eight GPUs, global batch size 16, base seed 666,
`RV_VOX_CHUNK=512`, and FP32:

```bash
(cd tools && bash scripts/dist_train.sh 8 \
    --cfg_file cfgs/argoverse_models/rv_sdtm.yaml \
    --batch_size 16 --epochs 12 --workers 1 \
    --fix_random_seed --skip_post_eval \
    --extra_tag av2_e12_seed666 \
    --set MODEL.BACKBONE_3D.RV_VOX_CHUNK 512)
```

All canonical configurations set `USE_AMP: False` and `SEED: 666`.
The trainer uses `666 + global rank`; `--seed` overrides the base value.
Waymo ablation launches use the same command with a configuration from
`cfgs/waymo_models/ablations/` and a distinct `--extra_tag`.

## Evaluation

The learned Router uses deterministic top-score selection over its full budget
at evaluation time. Use the following fixed-seed evaluation command:

```bash
python tools/test.py \
    --cfg_file cfgs/nuscenes_models/rv_sdtm.yaml \
    --ckpt /path/to/checkpoint_epoch_24.pth \
    --fix_random_seed --eval_seed 666 \
    --root_dir . --output_dir output/nuscenes_rv_sdtm_eval
```

Evaluation requires a matching model configuration and checkpoint. Use
`--ckpt` for full training resume, including optimizer state, or
`--pretrained_model` for partial model-only initialization. See
[checkpoint loading](RV_SDTM_RELEASE.md#checkpoint-loading).

For AV2 evaluation, run on four GPUs with global batch size 8 and seed 666:

```bash
(cd tools && bash scripts/dist_test.sh 4 \
    --cfg_file cfgs/argoverse_models/rv_sdtm.yaml \
    --ckpt /path/to/checkpoint_epoch_12.pth \
    --batch_size 8 --workers 1 \
    --fix_random_seed --eval_seed 666 \
    --root_dir .. --output_dir ../output/av2_rv_sdtm_eval \
    --set MODEL.BACKBONE_3D.RV_VOX_CHUNK 512)
```

AV2 evaluates all 26 categories under the 200 m ROI-only protocol, configured
by `DATA_CONFIG.EVALUATE_RANGE: 200.0` and
`DATA_CONFIG.EVAL_ONLY_ROI_INSTANCES: True`.

## Dataset-free verification

```bash
python tools/validate_rv_sdtm_release.py
python tools/validate_rv_sdtm_release.py --build
python tools/validate_rv_sdtm_release.py --argo2-forward
python tools/test_rv_sdtm_method.py
python -m compileall -q pcdet tools
```

Model construction and sparse forward/backward checks require a CUDA device
and compiled extensions. These checks use generated inputs without reading
dataset files.
