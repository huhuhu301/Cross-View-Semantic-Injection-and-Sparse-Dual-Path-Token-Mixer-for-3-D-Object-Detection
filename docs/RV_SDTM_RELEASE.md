# RV-SDTM implementation guide

RV-SDTM is implemented on OpenPCDet. See [Installation](INSTALL.md) for the
environment and [Getting started](GETTING_STARTED.md) for dataset preparation.

## Model paths

| Dataset | VFE | Backbone | Detection head |
|---|---|---|---|
| Waymo | RangeViewVFE | RVSDTM | SparseDynamicHead |
| nuScenes | RangeViewVFE | RVSDTMNuScenes | TransFusionHead |
| Argoverse 2 | RangeViewVFE | RVSDTMAV2 | SparseDynamicHead |

Canonical YAML files are under `cfgs/{waymo,nuscenes,argoverse}_models/`.
Use their `tools/cfgs/` counterparts when launching from `tools/`.

## Range-view and voxel features

Each occupied RV pixel averages six geometric attributes:
`(x, y, z, range, azimuth, elevation)`. The RV encoder produces 64 channels.
Integer pixel coordinates gather RV features for each point. These features
are concatenated with point attributes, the cluster offset and the voxel-center
offset before PFN aggregation. Waymo uses `5 + 3 + 3 + 64 = 75` PFN input channels
and 64 output channels.

Voxel anchors are arithmetic means of continuous normalized horizontal and
vertical projection coordinates. RV token anchors use occupied integer pixels
divided by `W-1` and `H-1`. Horizontal wrap-around is applied to candidate
distances, not to anchor averaging.

GeoCVA is applied once, after VFE and before the sparse stem. For Waymo:

- RV map: `64 × 1024`, vertical field of view `[-22.5°, 22.5°]`.
- Voxel features: 64 channels; RV pre-projection: `64 → 80`.
- Q/K/V: 80 channels, four heads of 20 channels; output projection: `80 → 64`.
- Angular neighbors: `K=6`; cap: 12,000 RV tokens per sample.
- Angular scales: `su=0.05`, `sv=0.08`; `sigma=tau=1`.
- Residual injection: fixed scalar `alpha=0.3`, followed by LayerNorm.

With wrapped angular distance `d`, logits are
`QKᵀ / (sqrt(head_dim) * tau) - d² / (2*sigma²)`.
Softmax is over the selected neighbors. The concatenated multi-head message
passes through the output projection before residual addition and LayerNorm.

For `M` occupied tokens and cap `C`, sampling retains
`arange(0, M, ceil(M/C))[:C]` when `M>C`, and all tokens otherwise.
This deterministic rule is shared by training, inference and all fusion variants.
Point-level RV gathering uses the complete feature map.

## Sparse dual-path mixer

The Waymo stem contains an input sparse convolution, three sparse residual
blocks and a height downsampling convolution with stride `(2,1,1)`.
Stage 1 contains one SDTM block without Router, followed by stride-`(2,2,2)`
downsampling. Stage 2 contains one SDTM block with Router.

Local Mixer operators are:

```text
SubMConv 3×3×3 → BN → SiLU
→ SubMConv 1×3×3 → BN → SiLU
→ SubMConv 1×1×1 → BN
```

All three layers preserve channels. The three BatchNorm modules have independent
parameters, with `eps=1e-3` and `momentum=0.01`. There is no final activation.

Local Mixer, Router downsampling and the fusion gate all receive the original
block input `x`. The block output is
`x + Local(x) + sigmoid(Linear(x)) * Router(x)`.
Blocks without Router return `x + Local(x)`.

Waymo Router uses 48 coarse channels, stride `(1,8,8)`, and two attention heads
of 24 channels. For each sample with `N` coarse tokens, its budget is
`min(N, max(2, ceil(0.3*N)))`. During training, the scoring MLP adds Gaussian
noise with standard deviation 0.1; up to
`min(B-1, round(0.25*B))` budget slots are sampled uniformly from the remaining
tokens after top-score selection. Python's round-to-even rule applies.
Evaluation uses all `B` slots for top-score selection without noise.

The selected features use the identity-forward score-gradient path
`z * (1 + (sigmoid(score(z)/0.8) - sigmoid(score(z)/0.8).detach()))`.
Linear attention increments are scattered to selected coarse rows and added
to the complete coarse tensor; unselected rows retain their original features.
Matched-index inverse convolution restores the input active sites and channels.

## Waymo ablations

All variants inherit the same Waymo training schedule and detection head.

| Variant | Configuration suffix | Change |
|---|---|---|
| Voxel-only | `voxel_only` | Remove RV encoder, point-level gathering and voxel-level fusion |
| RV + Add | `add` | Add a bias-free projection of the nearest angular RV token |
| RV + Concat | `concat` | Project concatenated voxel and nearest angular RV token; no outer residual |
| GeoCVA | canonical `rv_sdtm.yaml` | Angular multi-head cross-attention |
| Range-aware GeoCVA | `geocva_range` | Add relative radial bias to the same angular neighbors |
| w/o GeoCVA | `wo_geocva` | Keep point-level RV fusion; remove voxel-level fusion |
| Local-only | `local_only` | Disable Router and its gate |
| Router-only | `router_only` | Disable Local Mixer and the fusion gate |
| w/o Gate | `wo_gate` | Sum local and global increments with unit global weight |
| z-max pooling | `z_maxpool` | Replace attentional pillar pooling with height max pooling |
| w/o AFD | `wo_afd` | Remove adaptive feature diffusion |

Files are named `rv_sdtm_<suffix>.yaml` under `cfgs/waymo_models/ablations/`
and `tools/cfgs/waymo_models/ablations/`.

Add and Concat share the GeoCVA RV cap, `64 → 80` RV pre-projection and angular
distance. Add uses `x + Linear80→64(rv)`; Concat uses
`Linear144→64(concat(x,rv))`. Neither applies LayerNorm or an activation.

Range-aware GeoCVA adds
`-lambda * (abs(r_voxel-r_rv) / max(r_voxel,r_rv,eps))² / (2*sigma_r²)`
to the attention logits, after QK temperature scaling. Ranges are means of
3-D point distances within each voxel or RV pixel.
Defaults are `lambda=1`, `sigma_r=0.2` and `eps=1e-6`.

Example:

```bash
(cd tools && bash scripts/dist_train.sh 4 \
    --cfg_file cfgs/waymo_models/ablations/rv_sdtm_concat.yaml \
    --batch_size 8 --epochs 12 --workers 2 \
    --fix_random_seed --sync_bn --skip_post_eval \
    --extra_tag concat_e12_seed666)
```

## Training defaults

Waymo uses 12 epochs, global batch 8 in the four-GPU launch, FP32,
synchronized BatchNorm, and base seed 666. Each rank uses `666 + global rank`.
The optimizer is `OptimWrapper(torch.optim.Adam)` with decoupled weight decay
0.05 and a cosine OneCycle schedule: peak LR 0.003, warm-up fraction 0.1,
division factor 100. `--seed`, `--batch_size` and `--epochs` override defaults.

nuScenes uses 24 epochs and AV2 uses 12 epochs, both with global batch 16 in
the eight-GPU commands in [README](../README.md#3-reproduce-training).

## Distance-stratified evaluation

For Waymo distance-stratified comparisons, filter predictions and ground truth
by the same 3-D box-center distance before invoking the official evaluator:

```bash
python tools/eval_waymo_ranges.py \
    --predictions output/waymo_eval/result.bin \
    --ground-truth data/waymo/gt.bin \
    --metrics-binary /path/to/compute_detection_metrics_main \
    --output-dir output/waymo_distance_eval
```

This evaluates the five intervals `[0,15)`, `[15,30)`, `[30,45)`,
`[45,60)`, `[60,75)` m and the additional `[50,75)` m interval separately.
JSON scores are fractions in `[0,1]`; multiply by 100 for percentages.
Each interval retains the official per-class L1/L2 AP and APH output.

## Checkpoint loading

Evaluate or resume with a checkpoint produced by the same architecture and
configuration. In particular, attention dimensions, fusion variant and
detection head must match. Evaluation checks that all model tensors load;
full resume additionally restores matching optimizer state.

Use `--pretrained_model` for partial, model-only initialization and `--ckpt`
for full resume. Model-only initialization logs all unmatched tensors.

## Inference efficiency

Use one GPU, batch size 1 and FP32 for complete-detector measurements:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/profile_inference.py \
    --cfg_file cfgs/waymo_models/rv_sdtm.yaml \
    --ckpt /path/to/checkpoint_epoch_12.pth \
    --warmup 100 --frames 1000 --runs 3 \
    --output output/waymo_runtime.json
```

The timer synchronizes CUDA before and after each forward pass. It includes
model-side voxelization, RV processing, backbone, detection head, decoding and
NMS; it excludes data loading, host-to-device transfer and metric calculation.
The report records the GPU model, software versions, parameter count, per-run
mean latency and peak allocated memory. Use the same GPU and settings for all
compared configurations.

## Dataset-free checks

```bash
python tools/validate_rv_sdtm_release.py
python tools/validate_rv_sdtm_release.py --build --argo2-forward
python tools/test_rv_sdtm_method.py
python -m compileall -q pcdet tools
```

The model construction and sparse forward/backward checks require a CUDA device
and compiled extensions. They use generated inputs, not dataset evaluation.
