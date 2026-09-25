# Modified by RV-SDTM contributors for this public release; see NOTICE.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform_subsample_indices(count, cap, device):
    if cap < 1:
        raise ValueError('RV token cap must be positive')
    stride = max(1, math.ceil(count / cap))
    return torch.arange(0, count, stride, device=device)[:cap]


def wrapped_angular_distance(vox_uv, rv_uv, su, sv):
    du = torch.abs(vox_uv[:, None, 0] - rv_uv[None, :, 0])
    du = torch.minimum(du, 1.0 - du)
    dv = torch.abs(vox_uv[:, None, 1] - rv_uv[None, :, 1])
    return torch.sqrt((du / su) ** 2 + (dv / sv) ** 2)


def normalized_range_bias(vox_ranges, rv_ranges, weight=1.0, sigma=0.2, eps=1e-6):
    denominator = torch.maximum(vox_ranges, rv_ranges).clamp_min(eps)
    relative_difference = (vox_ranges - rv_ranges).abs() / denominator
    return -weight * relative_difference.square() / (2 * sigma ** 2)


class CrossViewDCA(nn.Module):
    """
    Range-View guided cross-view attention for voxel features.

    Args:
        vox_c: voxel feature channel dim (input to backbone, e.g., 64)
        rv_c:  range-view token channel dim (after optional projection)
        attn_dim: hidden dim for attention (divisible by heads)
        heads: number of attention heads
        k: top-k nearest RV tokens (by wrapped UV distance) per voxel
        max_rv_tokens: upper bound of RV tokens per frame (uniform subsampling)
        vox_chunk: process voxels by chunks to limit memory
        su, sv: UV distance scaling for wrap-aware distance
        sigma: Gaussian bias width on distance
        tau: temperature for attention logits
        alpha_init: residual injection coefficient
    """

    def __init__(self, vox_c: int, rv_c: int, attn_dim: int = 128, heads: int = 8,
                 k: int = 16, max_rv_tokens: int = 12000, vox_chunk: int = 2048,
                 su: float = 0.05, sv: float = 0.08, sigma: float = 1.0, tau: float = 1.0,
                 alpha_init: float = 0.3, learnable_alpha: bool = True,
                 use_range_bias: bool = False, range_weight: float = 1.0,
                 range_sigma: float = 0.2, range_eps: float = 1e-6):
        super().__init__()
        if attn_dim % heads != 0:
            raise ValueError(f"attn_dim ({attn_dim}) must be divisible by heads ({heads})")
        self.k, self.heads = k, heads
        self.max_rv_tokens, self.vox_chunk = max_rv_tokens, vox_chunk
        self.attn_dim = attn_dim
        self.su, self.sv, self.sigma = su, sv, sigma
        self.tau = tau
        if min(k, max_rv_tokens, vox_chunk, heads, su, sv, sigma, tau, range_sigma, range_eps) <= 0:
            raise ValueError('GeoCVA dimensions, budgets and scales must be positive')
        self.use_range_bias = use_range_bias
        self.range_weight, self.range_sigma, self.range_eps = range_weight, range_sigma, range_eps

        self.ln_q = nn.LayerNorm(vox_c)
        self.ln_kv = nn.LayerNorm(rv_c)

        self.wq = nn.Linear(vox_c, attn_dim, bias=False)
        self.wk = nn.Linear(rv_c, attn_dim, bias=False)
        self.wv = nn.Linear(rv_c, attn_dim, bias=False)

        self.wo = nn.Linear(attn_dim, vox_c, bias=False)
        self.norm = nn.LayerNorm(vox_c)

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer('alpha', torch.tensor(float(alpha_init)))

    def _subsample_uniform(self, rv_feat, rv_uv):
        idx = uniform_subsample_indices(rv_feat.size(0), self.max_rv_tokens, rv_feat.device)
        return rv_feat[idx], rv_uv[idx]

    def _wrap_knn_dist(self, vox_uv_i, rv_uv_all):
        return wrapped_angular_distance(vox_uv_i, rv_uv_all, self.su, self.sv)

    def forward(self, vox_feat, vox_uv, rv_feat, rv_uv, vox_ranges=None, rv_ranges=None):
        if rv_feat.numel() == 0 or vox_feat.numel() == 0:
            return vox_feat

        output_dtype = vox_feat.dtype
        d = self.attn_dim
        heads = self.heads
        head_dim = d // heads

        keep = uniform_subsample_indices(rv_feat.size(0), self.max_rv_tokens, rv_feat.device)
        rv_feat, rv_uv = rv_feat[keep], rv_uv[keep]
        if self.use_range_bias:
            if vox_ranges is None or rv_ranges is None:
                raise ValueError('Range-aware GeoCVA requires voxel and RV mean 3-D ranges')
            vox_ranges, rv_ranges = vox_ranges.float(), rv_ranges[keep].float()

        # Attention and residual normalization accumulate in FP32.
        with torch.cuda.amp.autocast(enabled=False):
            vox_feat_fp32 = vox_feat.float()
            vox_uv_fp32 = vox_uv.float()
            rv_feat_fp32 = rv_feat.float()
            rv_uv_fp32 = rv_uv.float()

            q_all = self.wq(self.ln_q(vox_feat_fp32)).reshape(
                -1, heads, head_dim
            )
            kv_in = self.ln_kv(rv_feat_fp32)
            k_all = self.wk(kv_in)
            v_all = self.wv(kv_in)

            out = vox_feat_fp32.clone()
            num_vox, num_rv = vox_feat_fp32.size(0), rv_feat_fp32.size(0)

            for s in range(0, num_vox, self.vox_chunk):
                e = min(s + self.vox_chunk, num_vox)
                uv_i = vox_uv_fp32[s:e]
                q_i = q_all[s:e]
                ni = q_i.size(0)

                dist = self._wrap_knn_dist(uv_i, rv_uv_fp32)
                k_use = min(self.k, num_rv)
                _, idx = torch.topk(dist, k=k_use, dim=1, largest=False)

                flat_idx = idx.reshape(-1)
                k_sel = k_all.index_select(0, flat_idx).reshape(
                    ni, k_use, d
                ).reshape(ni, k_use, heads, head_dim)
                v_sel = v_all.index_select(0, flat_idx).reshape(
                    ni, k_use, d
                ).reshape(ni, k_use, heads, head_dim)
                k_sel = k_sel.transpose(1, 2).contiguous()
                v_sel = v_sel.transpose(1, 2).contiguous()

                dist_sel = dist.gather(1, idx)
                dist_bias = -(dist_sel ** 2) / (2 * (self.sigma ** 2))
                if self.use_range_bias:
                    dist_bias = dist_bias + normalized_range_bias(
                        vox_ranges[s:e, None], rv_ranges[idx],
                        self.range_weight, self.range_sigma, self.range_eps,
                    )

                attn_logits = (
                    (q_i.unsqueeze(2) * k_sel).sum(-1)
                    / (math.sqrt(head_dim) * max(self.tau, 1e-6))
                )
                attn = F.softmax(
                    attn_logits + dist_bias[:, None, :], dim=-1
                )
                message = (
                    attn.unsqueeze(-1) * v_sel
                ).sum(-2).reshape(ni, d)

                out[s:e] = self.norm(
                    vox_feat_fp32[s:e]
                    + self.alpha.clamp(0.0, 1.0) * self.wo(message)
                )

        return out.to(output_dtype)


class AngularFusion(nn.Module):
    """Nearest-angular-token addition or concatenation without normalization."""

    def __init__(self, vox_c, rv_c, mode, max_rv_tokens=12000, vox_chunk=2048,
                 su=0.05, sv=0.08):
        super().__init__()
        if mode not in ('add', 'concat'):
            raise ValueError('AngularFusion mode must be add or concat')
        if min(max_rv_tokens, vox_chunk, su, sv) <= 0:
            raise ValueError('Fusion budgets and angular scales must be positive')
        self.mode = mode
        self.max_rv_tokens, self.vox_chunk = max_rv_tokens, vox_chunk
        self.su, self.sv = su, sv
        self.projection = nn.Linear(rv_c if mode == 'add' else vox_c + rv_c, vox_c, bias=False)

    def forward(self, vox_feat, vox_uv, rv_feat, rv_uv, **kwargs):
        if vox_feat.numel() == 0 or rv_feat.numel() == 0:
            return vox_feat
        keep = uniform_subsample_indices(rv_feat.size(0), self.max_rv_tokens, rv_feat.device)
        with torch.cuda.amp.autocast(enabled=False):
            rv_feat, rv_uv = rv_feat[keep].float(), rv_uv[keep].float()
            output = []
            for start in range(0, vox_feat.size(0), self.vox_chunk):
                end = start + self.vox_chunk
                nearest = wrapped_angular_distance(vox_uv[start:end].float(), rv_uv, self.su, self.sv).argmin(1)
                voxel, message = vox_feat[start:end].float(), rv_feat[nearest]
                if self.mode == 'add':
                    output.append(voxel + self.projection(message))
                else:
                    output.append(self.projection(torch.cat([voxel, message], dim=-1)))
        return torch.cat(output, dim=0).to(vox_feat.dtype)


def build_cross_view_fusion(config, voxel_channels):
    mode = config.get('RV_FUSION', 'geocva' if config.get('USE_RV', False) else 'none')
    if mode == 'none':
        return None, None
    if mode not in ('geocva', 'geocva_range', 'add', 'concat'):
        raise ValueError('Unsupported RV_FUSION: {}'.format(mode))
    rv_channels = int(config['RV_FEATURE_DIM'])
    dim = int(config['RV_ATTN_DIM'])
    projection = nn.Linear(rv_channels, dim, bias=False) if rv_channels != dim else nn.Identity()
    common = dict(vox_c=voxel_channels, rv_c=dim,
                  max_rv_tokens=config.get('RV_MAX_TOKENS', 12000),
                  vox_chunk=config.get('RV_VOX_CHUNK', 2048),
                  su=config.get('RV_SU', 0.05), sv=config.get('RV_SV', 0.08))
    if mode in ('add', 'concat'):
        aggregation = AngularFusion(mode=mode, **common)
    else:
        aggregation = CrossViewDCA(
            attn_dim=dim, heads=config.get('RV_HEADS', 8), k=config.get('RV_DCA_K', 6),
            sigma=config.get('RV_SIGMA', 1.0), tau=config.get('RV_TAU', 1.0),
            alpha_init=config.get('RV_ALPHA_INIT', 0.3),
            learnable_alpha=config.get('RV_ALPHA_LEARNABLE', True),
            use_range_bias=(mode == 'geocva_range'),
            range_weight=config.get('RV_RANGE_WEIGHT', 1.0),
            range_sigma=config.get('RV_RANGE_SIGMA', 0.2),
            range_eps=config.get('RV_RANGE_EPS', 1e-6), **common)
    return projection, aggregation


def fuse_rv_features(features, batch_ids, batch_dict, projection, aggregation):
    if aggregation is None:
        return features
    for key in ('rv_tokens', 'rv_uv', 'vox_uv'):
        if key not in batch_dict:
            raise KeyError('Cross-view fusion requires {}'.format(key))
    anchors = batch_dict['vox_uv']
    if anchors.shape != (features.shape[0], 2):
        raise ValueError('Voxel angular anchors must be row-aligned with voxel features')
    output = features.clone()
    use_range = getattr(aggregation, 'use_range_bias', False)
    if use_range and not all(key in batch_dict for key in ('vox_ranges', 'rv_ranges')):
        raise KeyError('Range-aware fusion requires vox_ranges and rv_ranges')
    for b, tokens in enumerate(batch_dict['rv_tokens']):
        mask = batch_ids == b
        if not mask.any() or tokens.numel() == 0:
            continue
        ranges = dict(vox_ranges=batch_dict['vox_ranges'][mask],
                      rv_ranges=batch_dict['rv_ranges'][b]) if use_range else {}
        output[mask] = aggregation(features[mask], anchors[mask], projection(tokens),
                                   batch_dict['rv_uv'][b], **ranges)
    return output
