# Modified by RV-SDTM contributors; see NOTICE and THIRD_PARTY_NOTICES.md.
"""Sparse token mixing and geometric cross-view attention for KITTI."""
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.spconv_utils import spconv, replace_feature

norm_fn_1d = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
norm_fn_2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)

class LocalMixer(spconv.SparseModule):

    def __init__(self, dim, indice_tag):
        super().__init__()
        self.block = spconv.SparseSequential(
            spconv.SubMConv3d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, indice_key=f'{indice_tag}_lm1'),
            norm_fn_1d(dim),
            nn.SiLU(inplace=True),
            spconv.SubMConv3d(dim, dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False, indice_key=f'{indice_tag}_lm2'),
            norm_fn_1d(dim),
            nn.SiLU(inplace=True),
            spconv.SubMConv3d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False, indice_key=f'{indice_tag}_lm3'),
            norm_fn_1d(dim)
        )

    def forward(self, x):
        return self.block(x)


class RouterLinearAttention(nn.Module):

    def __init__(self, dim, num_heads=4, attn_dim=None, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn_dim = attn_dim if attn_dim is not None else dim
        assert self.attn_dim % num_heads == 0
        self.head_dim = self.attn_dim // num_heads
        self.eps = eps

        self.q_proj = nn.Linear(dim, self.attn_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.attn_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.attn_dim, bias=False)
        self.out_proj = nn.Linear(self.attn_dim, dim, bias=False)

    @staticmethod
    def kernel(x):
        return F.elu(x, alpha=1.0, inplace=False) + 1.0

    def forward(self, feats, batch_ids, batch_size):
        if feats.numel() == 0:
            return feats

        input_dtype = feats.dtype

        # Projections and attention reductions accumulate in FP32.
        with torch.cuda.amp.autocast(enabled=False):
            feats_fp32 = feats.float()
            q = self.q_proj(feats_fp32).view(
                -1, self.num_heads, self.head_dim
            )
            k = self.k_proj(feats_fp32).view(
                -1, self.num_heads, self.head_dim
            )
            v = self.v_proj(feats_fp32).view(
                -1, self.num_heads, self.head_dim
            )
            q_phi = self.kernel(q)
            k_phi = self.kernel(k)

            kv = torch.zeros(
                batch_size, self.num_heads, self.head_dim, self.head_dim,
                dtype=torch.float32, device=feats.device,
            )
            k_sum = torch.zeros(
                batch_size, self.num_heads, self.head_dim,
                dtype=torch.float32, device=feats.device,
            )

            for b in range(batch_size):
                mask = batch_ids == b
                if not mask.any():
                    continue
                kb = k_phi[mask]
                vb = v[mask]
                kv[b] = torch.einsum('nhd,nhe->hde', kb, vb)
                k_sum[b] = kb.sum(dim=0)

            # Apply each sample's global statistics to its selected tokens.
            batch_ids = batch_ids.long()
            kv_per_token = kv.index_select(0, batch_ids)
            k_sum_per_token = k_sum.index_select(0, batch_ids)
            numer = torch.einsum('nhd,nhde->nhe', q_phi, kv_per_token)
            denom = (
                torch.einsum('nhd,nhd->nh', q_phi, k_sum_per_token)
                + self.eps
            )
            out = numer / denom.unsqueeze(-1)
            out = out.reshape(feats.shape[0], self.attn_dim)
            out = self.out_proj(out)

        return out.to(input_dtype)


class RouterBranch(spconv.SparseModule):

    def __init__(self, dim, router_dim, stride_xy, attn_heads, attn_dim, indice_tag,
                 keep_ratio=1.0, selection_strategy='importance', selection_cfg=None):
        super().__init__()
        stride = [1, stride_xy, stride_xy]
        self.router_down = spconv.SparseSequential(
            spconv.SparseConv3d(dim, router_dim, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=f'{indice_tag}_down'),
            norm_fn_1d(router_dim),
            nn.SiLU(inplace=True)
        )
        self.router_up = spconv.SparseSequential(
            spconv.SparseInverseConv3d(router_dim, dim, kernel_size=3, indice_key=f'{indice_tag}_down'),
            norm_fn_1d(dim)
        )
        self.attn = RouterLinearAttention(router_dim, num_heads=attn_heads, attn_dim=attn_dim)
        self.keep_ratio = keep_ratio
        self.selection_strategy = selection_strategy
        self.selection_cfg = selection_cfg or {}
        self.min_tokens = max(1, int(self.selection_cfg.get('MIN_TOKENS', 1)))
        self.hybrid_random_ratio = float(self.selection_cfg.get('HYBRID_RANDOM_RATIO', 0.3))
        self._sampling_generator = None
        self.score_temperature = float(self.selection_cfg.get('SCORE_TEMPERATURE', 1.0))
        self.score_noise = float(self.selection_cfg.get('SCORE_NOISE_STD', 0.0))
        if self.score_temperature <= 0 or not 0 < self.keep_ratio <= 1:
            raise ValueError('Router temperature must be positive and keep ratio must be in (0, 1]')
        self.score_straight_through = bool(
            self.selection_cfg.get('STRAIGHT_THROUGH_SCORE', True)
        )
        if self.selection_strategy in ['learned', 'learned_hybrid']:
            hidden = int(self.selection_cfg.get('SCORE_HIDDEN_DIM', max(router_dim // 2, 32)))
            self.score_head = nn.Sequential(
                nn.Linear(router_dim, hidden),
                nn.SiLU(inplace=True),
                nn.Linear(hidden, 1)
            )
        else:
            self.score_head = None

    def forward(self, x):
        router = self.router_down(x)
        if router.features.numel() == 0:
            return spconv.SparseConvTensor(
                features=x.features.new_zeros(x.features.shape),
                indices=x.indices,
                spatial_shape=x.spatial_shape,
                batch_size=x.batch_size
            )

        features = router.features
        indices = router.indices
        keep_mask = None
        if self.keep_ratio < 1.0:
            batch_ids = indices[:, 0]
            keep_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
            for b in range(router.batch_size):
                b_mask = batch_ids == b
                if not b_mask.any():
                    continue
                b_indices = b_mask.nonzero(as_tuple=True)[0]
                num = max(self.min_tokens, int(math.ceil(b_indices.numel() * self.keep_ratio)))
                selected = self._select_indices(features, b_indices, num)
                keep_mask[selected] = True
            features = features[keep_mask]
            sub_indices = indices[keep_mask]
        else:
            sub_indices = indices

        features = self._apply_score_gradient(features)
        attn_feat = self.attn(features, sub_indices[:, 0], router.batch_size)
        if keep_mask is not None:
            full_features = router.features.new_zeros(router.features.shape)
            full_features[keep_mask] = attn_feat
            attn_feat = full_features
        router = replace_feature(router, router.features + attn_feat)
        router = self.router_up(router)
        return router

    def _apply_score_gradient(self, features):
        if not self.score_straight_through or self.score_head is None:
            return features
        probability = torch.sigmoid(self.score_head(features).squeeze(-1) / self.score_temperature)
        scale = 1.0 + (probability - probability.detach())
        return features * scale.unsqueeze(-1)

    def _generator(self, device):
        # Keep router sampling independent of point sampling and dropout.
        if self._sampling_generator is None or self._sampling_generator.device != device:
            self._sampling_generator = torch.Generator(device=device)
            self._sampling_generator.manual_seed(torch.initial_seed() + 131071)
        return self._sampling_generator

    def _compute_scores(self, feats):
        strategy = self.selection_strategy
        if strategy in ['importance', 'hybrid']:
            return feats.pow(2).sum(dim=1)
        if strategy in ['learned', 'learned_hybrid'] and self.score_head is not None:
            scores = self.score_head(feats).squeeze(-1)
            if self.training and self.score_noise > 0:
                scores = scores + torch.randn(scores.shape, device=scores.device, dtype=scores.dtype, generator=self._generator(scores.device)) * self.score_noise
            if self.score_temperature != 1.0:
                scores = scores / self.score_temperature
            return scores
        return feats.new_ones(feats.shape[0])

    def _random_select(self, candidate_idx, num):
        if num <= 0 or candidate_idx.numel() == 0:
            return candidate_idx.new_empty(0, dtype=torch.long)
        if candidate_idx.numel() <= num:
            return candidate_idx
        perm = torch.randperm(candidate_idx.shape[0], device=candidate_idx.device, generator=self._generator(candidate_idx.device))
        return candidate_idx[perm[:num]]

    def _select_indices(self, features, candidate_idx, num_keep):
        if candidate_idx.numel() == 0:
            return candidate_idx
        num_keep = min(num_keep, candidate_idx.numel())
        if num_keep == candidate_idx.numel():
            return candidate_idx
        if self.selection_strategy == 'random' or candidate_idx.numel() == 1:
            return self._random_select(candidate_idx, num_keep)

        feats = features[candidate_idx]
        scores = self._compute_scores(feats)

        if not self.training or self.selection_strategy in ['importance', 'learned']:
            topk = torch.topk(scores, k=num_keep, dim=0).indices
            return candidate_idx[topk]

        if self.selection_strategy in ['hybrid', 'learned_hybrid']:
            random_ratio = min(max(self.hybrid_random_ratio, 0.0), 0.99)
            random_num = int(round(num_keep * random_ratio))
            random_num = min(num_keep - 1, random_num) if num_keep > 1 else 0
            topk_num = num_keep - random_num
            if topk_num <= 0:
                topk_num = 1
                random_num = num_keep - 1
            topk = torch.topk(scores, k=topk_num, dim=0).indices
            selected = candidate_idx[topk]
            if random_num > 0:
                mask = torch.ones(candidate_idx.shape[0], dtype=torch.bool, device=candidate_idx.device)
                mask[topk] = False
                remaining = candidate_idx[mask.nonzero(as_tuple=True)[0]]
                rand_sel = self._random_select(remaining, random_num)
                selected = torch.cat([selected, rand_sel], dim=0)
            return selected

        # default fallback
        return self._random_select(candidate_idx, num_keep)


class SDTMBlock(spconv.SparseModule):

    def __init__(self, dim, router_dim, stride_xy, attn_heads, attn_dim, indice_tag,
                 use_router=True, keep_ratio=1.0, selection_strategy='importance', selection_cfg=None,
                 use_local=True, use_gate=True):
        super().__init__()
        self.local_mixer = LocalMixer(dim, indice_tag) if use_local else None
        self.use_router = use_router
        self.use_gate = use_gate
        if use_router:
            self.router_branch = RouterBranch(
                dim, router_dim, stride_xy, attn_heads, attn_dim, indice_tag,
                keep_ratio=keep_ratio, selection_strategy=selection_strategy, selection_cfg=selection_cfg
            )
            if use_gate:
                self.router_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, x):
        residual = x
        if self.local_mixer is not None:
            lm = self.local_mixer(x)
            x = replace_feature(x, x.features + lm.features)
        if self.use_router:
            router = self.router_branch(residual)
            gate = self.router_gate(residual.features) if self.use_gate else 1.0
            fused = x.features + gate * router.features
            x = replace_feature(x, fused)
        return x


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


def post_act_block_dense_2d(input_dim, output_dim, kernel_size, stride, padding, dilation=1, groups=1, norm_fn=norm_fn_2d, conv_type='conv'):

    if conv_type == 'conv':
        conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, dilation, groups=groups, bias=False)

    elif conv_type == 'deconv':
        conv = nn.ConvTranspose2d(input_dim, output_dim, stride, stride, groups=groups, bias=False)

    else:
        raise NotImplementedError

    return nn.Sequential(conv, norm_fn(output_dim), nn.ReLU())


class BasicBlock(nn.Module):

    def __init__(self, dim, groups=1, norm_fn=norm_fn_2d, downsample=False):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, 3, 1 + int(downsample), 1, groups=groups, bias=False)
        self.bn1 = norm_fn(dim)

        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, groups=groups, bias=False)
        self.bn2 = norm_fn(dim)
        self.relu = nn.ReLU()

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 2, 0, groups=groups, bias=False),
                norm_fn(dim),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if hasattr(self, 'downsample'):
            x = self.downsample(x)

        out = self.relu(out + x)
        return out


class DEDLayer(nn.Module):
    """Multi-level dense encoder-decoder block with additive skip fusion."""

    def __init__(self, dim: int, down_stride: list, num_SBB: list):
        super().__init__()

        self.encoder = nn.ModuleList(
            nn.Sequential(*[BasicBlock(dim) for _ in range(num_SBB[0])])
        )

        num_levels = len(down_stride)
        for idx in range(1, num_levels):
            cur_layers = [BasicBlock(dim, downsample=True)]
            cur_layers.extend(
                [BasicBlock(dim) for _ in range(num_SBB[idx])]
            )
            self.encoder.append(nn.Sequential(*cur_layers))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            self.decoder.append(
                post_act_block_dense_2d(
                    dim,
                    dim,
                    down_stride[idx],
                    down_stride[idx],
                    0,
                    conv_type='deconv',
                )
            )
            self.decoder_norm.append(norm_fn_2d(dim))

    def forward(self, x):
        features = []
        for encoder in self.encoder:
            x = encoder(x)
            features.append(x)

        for decoder, norm, skip in zip(
                self.decoder, self.decoder_norm, features[:-1][::-1]):
            x = norm(decoder(x) + skip)

        return x
