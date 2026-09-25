# Installation

This guide covers the validated environment for RV-SDTM. Run installation
commands from the cloned repository root.

## Hardware validation

The project has completed training and inference validation on the following
GPU platforms:

| GPU | Validated workflow |
|---|---|
| NVIDIA A10 | Training, evaluation, and release build checks |
| NVIDIA A100 | Training |
| NVIDIA GeForce RTX 4090 | Inference and runtime profiling |

These validations establish compatibility for the corresponding workflows.
The version-pinned installation instructions below document the **A10
reference environment**; the hardware table records the broader validation
coverage of the project.

## Recommended and validated stack

Use the local A10-server stack that produced the archived AV2 result and passed
the release build and synthetic validation:

- Ubuntu 18.04
- Python 3.8.20
- PyTorch 1.10.0+cu113
- torchvision 0.11.0+cu113
- CUDA Toolkit 11.3 (`nvcc` 11.3.58)
- cuDNN 8.2
- spconv-cu113 2.3.6
- torch-scatter 2.1.2
- NVIDIA driver 470.82.01 and NVIDIA A10 24 GB GPUs

Install PyTorch from its matching CUDA wheel index before installing the
repository dependencies:

```bash
conda create -n rv-sdtm python=3.8.20
conda activate rv-sdtm

export CUDA_HOME=/usr/local/cuda-11.3
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

`torchaudio` is not imported by this repository and was not installed in the
validated local environment.

`requirements.txt` includes the core packages and all three dataset-specific
files. Install it for an environment that will use every benchmark. The
dataset registry loads adapters lazily, so a single-dataset environment can
instead install `requirements-core.txt` plus exactly one matching file:

- `requirements-core.txt`
- `requirements-waymo.txt`
- `requirements-nuscenes.txt`
- `requirements-av2.txt`

## Build CUDA extensions

Compile and install `pcdet` in editable mode from the repository root:

```bash
python setup.py develop
```

The public build compiles only the two CUDA extensions used by the three
canonical protocols: `iou3d_nms` and `roiaware_pool3d`.

The distribution name is `rv-sdtm-pcdet`; the Python import namespace remains
`pcdet` for configuration and checkpoint compatibility.

Prebuilt `.so` files are intentionally excluded because they are tied to a
specific Python, PyTorch, CUDA, compiler, and GPU architecture combination.

## Verify the installation

Run the dataset-free checks:

```bash
python tools/validate_rv_sdtm_release.py
python tools/validate_rv_sdtm_release.py --build
python tools/validate_rv_sdtm_release.py --argo2-forward
python -m compileall pcdet tools
```

The last two validator modes require a CUDA device. Dataset SDKs may impose
additional constraints; use the pinned requirement files rather than an
unrelated environment freeze.

## Validation environment record

The stack above is the repository's recommended environment, not a legacy
fallback. The local installation additionally records NumPy 1.21.5, SciPy
1.9.1, AV2 0.2.1, kornia 0.6.0, and nuscenes-devkit 1.0.5. Release-level
configuration, CUDA-extension build, model construction, and synthetic
forward/backward checks were completed in this environment.

Changing only the environment documentation or rebuilding compatible CUDA
extensions does not require retraining an existing checkpoint. A fresh machine
should still record `python`, `torch`, `torch.version.cuda`, `spconv`, dataset
SDK versions, GPU model/count, config, seed, and checkpoint checksum when
publishing a reproduction. The Waymo SDK is only required for data conversion
or the official Python evaluator; the recorded version is 1.5.0 and is supplied
by `requirements-waymo.txt`.
