# Third-party notices

RV-SDTM is distributed under the Apache License 2.0 in the root `LICENSE`,
subject to retained upstream notices and any component-specific terms below.
This file is attribution information, not a replacement for those licenses.

## Source-code lineage included in this snapshot

- **OpenPCDet** provides the detector framework, dataset/evaluation substrate,
  training tools, and CUDA operations. RV-SDTM keeps the `pcdet` Python import
  namespace for compatibility. The upstream project is Apache-2.0 licensed.
- **FSHNet** provides the fully sparse hybrid detector foundation on which
  RV-SDTM is implemented. The upstream project is Apache-2.0 licensed.
- **HEDNet** is the source lineage for hierarchical sparse encoder-decoder
  utilities retained in this snapshot. The upstream project is Apache-2.0
  licensed.
- **OpenMMLab/MMDetection** is credited by the retained distributed-training
  initialization helper in `pcdet/utils/common_utils.py`. Preserve its
  source-level attribution and comply with the upstream Apache-2.0 license.
- **KITTI object evaluation Python code** is bundled beneath
  `pcdet/datasets/kitti/kitti_object_eval_python/` under the MIT License stored
  in that directory.

## External dependencies and datasets

PyTorch, spconv, torch-scatter, NumPy, SciPy, the Waymo Open Dataset package,
the nuScenes devkit, and the Argoverse 2 API are installed separately and are
not vendored by this repository. Each is governed by its own license.

Waymo Open Dataset, nuScenes, and Argoverse 2 data, annotations, evaluation
assets, and checkpoints are not redistributed. Users must obtain them from the
respective owners and comply with their dataset terms. Dataset and project
names remain trademarks of their respective owners; their mention does not
imply endorsement.
