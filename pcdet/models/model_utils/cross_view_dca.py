# Modified by RV-SDTM contributors for this public release; see NOTICE.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        alpha_init: initial value for residual gate
    """

    def __init__(self, vox_c: int, rv_c: int, attn_dim: int = 128, heads: int = 8,
                 k: int = 16, max_rv_tokens: int = 12000, vox_chunk: int = 2048,
                 su: float = 0.05, sv: float = 0.08, sigma: float = 1.0, tau: float = 1.0,
                 alpha_init: float = 0.3):
        super().__init__()
        if attn_dim % heads != 0:
            raise ValueError(f"attn_dim ({attn_dim}) must be divisible by heads ({heads})")
        self.k, self.heads = k, heads
        self.max_rv_tokens, self.vox_chunk = max_rv_tokens, vox_chunk
        self.attn_dim = attn_dim
        self.su, self.sv, self.sigma = su, sv, sigma
        self.tau = tau

        self.ln_q = nn.LayerNorm(vox_c)
        self.ln_kv = nn.LayerNorm(rv_c)

        self.wq = nn.Linear(vox_c, attn_dim, bias=False)
        self.wk = nn.Linear(rv_c, attn_dim, bias=False)
        self.wv = nn.Linear(rv_c, attn_dim, bias=False)

        self.wo = nn.Linear(attn_dim, vox_c, bias=False)
        self.norm = nn.LayerNorm(vox_c)

        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def _subsample_uniform(self, rv_feat, rv_uv):
        num = rv_feat.size(0)
        if num <= self.max_rv_tokens:
            return rv_feat, rv_uv
        stride = math.ceil(num / self.max_rv_tokens)
        idx = torch.arange(0, num, stride, device=rv_feat.device)[:self.max_rv_tokens]
        return rv_feat[idx], rv_uv[idx]

    def _wrap_knn_dist(self, vox_uv_i, rv_uv_all):
        u_i = vox_uv_i[:, 0:1]
        v_i = vox_uv_i[:, 1:2]
        u_r = rv_uv_all[None, :, 0]
        v_r = rv_uv_all[None, :, 1]

        du = torch.abs(u_i - u_r)
        du = torch.minimum(du, 1.0 - du)  # wrap around horizontally
        dv = torch.abs(v_i - v_r)
        dist = torch.sqrt((du / self.su) ** 2 + (dv / self.sv) ** 2)
        return dist

    def forward(self, vox_feat, vox_uv, rv_feat, rv_uv):
        if rv_feat.numel() == 0 or vox_feat.numel() == 0:
            return vox_feat

        output_dtype = vox_feat.dtype
        d = self.attn_dim
        heads = self.heads
        head_dim = d // heads

        rv_feat, rv_uv = self._subsample_uniform(rv_feat, rv_uv)

        # Keep the complete cross-view attention/residual island in FP32.
        # The module is parameter-compatible with existing checkpoints and
        # returns the incoming voxel dtype at its public boundary.
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
