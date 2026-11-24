# KITTI LiDAR object detection

The KITTI models detect Car, Pedestrian and Cyclist from a single LiDAR scan.
The combined model applies sparse dual-path token mixing to the fourth voxel
stage and uses range-view attention in each of three RoI classification stages.
The RoI box regressor keeps its original features.

## Configurations

All eight entries are standalone YAMLs. Training layouts below give GPU count
and frames per GPU. Single-module runs use the same training budget and RoI
refinement settings as the combined model.

| Configuration | Training GPUs × frames/GPU | Epochs |
|---|---:|---:|
| [GeoCVA + SDTM](../cfgs/kitti_models/kitti_rv_geocva_sdtm.yaml) | 4 × 1 | 100 |
| [GeoCVA only](../cfgs/kitti_models/kitti_rv_geocva.yaml) | 4 × 1 | 100 |
| [SDTM only](../cfgs/kitti_models/kitti_sdtm.yaml) | 4 × 1 | 100 |
| [PointPillars](../cfgs/kitti_models/kitti_pointpillars.yaml) | 4 × 4 | 80 |
| [SECOND](../cfgs/kitti_models/kitti_second.yaml) | 4 × 4 | 80 |
| [PointRCNN](../cfgs/kitti_models/kitti_pointrcnn.yaml) | 4 × 2 | 80 |
| [CT3D](../cfgs/kitti_models/kitti_ct3d.yaml) | 8 × 3 | 200 |
| [Part-A²-Anchor](../cfgs/kitti_models/kitti_parta2_anchor.yaml) | 4 × 4 | 80 |

## Data preparation

Install the [training environment and CUDA extensions](../README.md#1-environment-setup).
Download the LiDAR scans, left color images, camera calibration and labels from
the [KITTI object detection benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
Images supply the image shape for field-of-view filtering and box projection;
no image features enter the detector. Use the standard
[train/validation split lists](https://github.com/open-mmlab/OpenPCDet/tree/233f849829b6ac19afb8af8837a0246890908755/data/kitti/ImageSets)
(3,712/3,769 frames). The training configurations use road-plane augmentation:
extract the [road-plane files](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/docs/GETTING_STARTED.md#kitti-dataset)
under `training/planes/`, one file per training ID.

```text
data/kitti/
├── ImageSets/{train,val}.txt
└── training/
    ├── velodyne/
    ├── calib/
    ├── label_2/
    ├── image_2/
    └── planes/
```

From the repository root:

```bash
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \
    --cfg_file cfgs/dataset_configs/kitti_dataset.yaml \
    --data_path data/kitti --workers 4
```

Training uses `kitti_infos_train.pkl`, `kitti_dbinfos_train.pkl` and
`gt_database/`; validation uses `kitti_infos_val.pkl`. For data outside the
repository, pass the actual data path at preparation time and append
`--set DATA_CONFIG.DATA_PATH /absolute/path/to/kitti` to the training and
evaluation commands.

## Training

Use the [four-GPU launch command](../README.md#kitti-training). For the three
ablation configurations, use FP32, ordinary per-GPU BatchNorm, global batch 4,
and seed 666. The checkpoint used for validation is the fixed final epoch.
For an eight-GPU reference run across two four-GPU nodes, run the following on
each node from the same checkout, with the same shared KITTI and output paths.
Set `MASTER_ADDR` to the first node's reachable host name, `MASTER_PORT` to the
same available port, and `NODE_RANK` to 0 or 1:

```bash
(cd tools && torchrun --nnodes=2 --nproc_per_node=4 \
    --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    train.py --launcher pytorch --cfg_file cfgs/kitti_models/kitti_ct3d.yaml \
    --batch_size 24 --epochs 200 --workers 2 --fix_random_seed --seed 666 \
    --skip_post_eval --extra_tag kitti_s666)
```

## Validation metric

The native KITTI evaluator reports 3D AP_R40 at Easy / Moderate / Hard
difficulty (IoU 0.70 for Car, 0.50 for Pedestrian and Cyclist). The validation
loader uses the labeled split, not the unlabeled benchmark test server.

The metric can run in a separate interpreter with Numba's CUDA support. To
use the tested metric stack without changing the PyTorch training environment:

```bash
conda create -y -n rv-sdtm-kitti-metric python=3.11.5 numpy=1.24.3 numba=0.57.1 llvmlite=0.40.0
export KITTI_METRIC_PYTHON="$(conda run -n rv-sdtm-kitti-metric python -c 'import sys; print(sys.executable)')"
```

Keep the training environment active and run the
[four-GPU validation command](../README.md#kitti-validation). Supply the YAML
paired with the checkpoint. Predictions and the evaluation log are written
beneath the requested output directory; no external ground-truth binary is needed.
