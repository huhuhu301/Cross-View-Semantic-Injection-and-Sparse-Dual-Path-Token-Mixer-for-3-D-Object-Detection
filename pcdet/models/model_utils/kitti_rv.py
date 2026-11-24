# Modified by RV-SDTM contributors; see NOTICE and THIRD_PARTY_NOTICES.md.
"""RoI range-view injection and sparse backbone token mixing for KITTI."""
import math
import torch
from torch import nn

from .kitti_token_ops import CrossViewDCA, DEDLayer, SDTMBlock


def make_sparse(config, dim):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(config.INIT_SEED)
        block = SDTMBlock(dim, config.ROUTER_DIM, config.STRIDE_XY, config.HEADS,
                          config.ROUTER_DIM, 'kitti_sdtm4', use_router=True,
                          keep_ratio=config.KEEP, selection_strategy='learned_hybrid',
                          selection_cfg=dict(SCORE_HIDDEN_DIM=64, SCORE_TEMPERATURE=0.8,
                                             SCORE_NOISE_STD=0.1, STRAIGHT_THROUGH_SCORE=True,
                                             HYBRID_RANDOM_RATIO=0.25, MIN_TOKENS=2))
        # The trained residuals can grow, but step zero is an identity.
        nn.init.zeros_(block.local_mixer.block[-1].weight)
        nn.init.zeros_(block.router_branch.router_up[-1].weight)
    return block


class RVTokens(nn.Module):
    """Measured-ray projection, occupied pixels only, one lightweight DED pyramid."""
    def __init__(self, config):
        super().__init__()
        self.height, self.width = config.SHAPE
        self.rv_elevation_bounds = list(config.ELEVATION_DEG)
        width = config.CHANNELS
        self.stem = nn.Sequential(nn.Conv2d(6, width, 3, 1, 1), nn.BatchNorm2d(width), nn.ReLU())
        self.pyramid = DEDLayer(width, [1, 2], [1, 1])
        self.projection = nn.Linear(width, config.ATTN_DIM, bias=False)

    def forward(self, points, batch_size):
        h, w = self.height, self.width
        image, _, counts, _, _ = self.create_range_view_feature(points, h, w, batch_size=batch_size)
        features = self.pyramid(self.stem(image)).permute(0, 2, 3, 1).reshape(batch_size, h*w, -1)
        output = []
        for b in range(batch_size):
            ix = (counts.view(batch_size, -1)[b] > 0).nonzero(as_tuple=True)[0]
            uv = torch.stack(((ix % w).float()/(w-1), (ix // w).float()/(h-1)), dim=1)
            output.append((self.projection(features[b, ix]), uv))
        return output

    def create_range_view_feature(self, points, H=128, W=1024, fov_vertical=180, batch_size=None):
        device = points.device
        batch_idx = points[:, 0].long()
        xyz = points[:, 1:4]
        r = torch.norm(xyz, dim=1)                                   # [N]
        phi = torch.atan2(xyz[:, 1], xyz[:, 0])                      # [-pi, pi]
        theta = torch.asin((xyz[:, 2] / r.clamp(min=1e-6)).clamp(-1, 1))

        # Continuous pixel coordinates.
        u_float = ((phi + math.pi) / (2 * math.pi)) * (W - 1)        # [0, W-1]
        if self.rv_elevation_bounds is None:
            half_fov = math.radians(fov_vertical) / 2
            v_float = ((theta + half_fov) / (2 * half_fov)) * (H - 1)
        else:
            low, high = map(math.radians, self.rv_elevation_bounds)
            v_float = ((theta - low) / (high - low)) * (H - 1)

        # Quantized coordinates for indexing and point counts.
        u = u_float.clamp(0, W - 1).long()
        v = v_float.clamp(0, H - 1).long()
        linear_indices = batch_idx * (H * W) + v * W + u             # [N]

        # Point count per pixel.
        B = int(batch_size) if batch_size is not None else (int(batch_idx.max().item()) + 1 if points.shape[0] else 1)
        pixel_counts = torch.zeros(B * H * W, dtype=torch.long, device=device)
        pixel_counts.scatter_add_(0, linear_indices, torch.ones_like(linear_indices, dtype=torch.long, device=device))

        # Mean per-pixel feature: xyz, range, azimuth, and elevation.
        feats = torch.cat([xyz, r[:, None], phi[:, None], theta[:, None]], dim=1).to(device)  # [N,6]
        feat_sum = torch.zeros(B * H * W, feats.shape[-1], device=device, dtype=feats.dtype)
        idx2d = linear_indices.unsqueeze(-1).expand(-1, feats.shape[-1])
        feat_sum.scatter_add_(0, idx2d, feats)
        rv_features = feat_sum / pixel_counts.clamp(min=1).float().unsqueeze(-1)             # [B*H*W, 6]
        rv_features = rv_features.view(B, H, W, feats.shape[-1]).permute(0, 3, 1, 2).contiguous()  # [B,6,H,W]

        return rv_features, linear_indices, pixel_counts, u_float, v_float


class CascadeRVClassification(nn.Module):
    """Use 512-D native features and the *current stage's* RoIs; never replace reg input."""
    def __init__(self, config, channels):
        super().__init__()
        self.encoder = RVTokens(config)
        self.attn = CrossViewDCA(channels, config.ATTN_DIM, attn_dim=config.ATTN_DIM,
            heads=config.HEADS, k=config.K, max_rv_tokens=config.CAP, vox_chunk=256,
            su=config.SU, sv=config.SV, sigma=1.0, tau=1.0, alpha_init=config.ALPHA,
            learnable_alpha=False, use_range_bias=False)
        # Preserve the native cls feature scale: the classification MLP already applies normalization.
        self.attn.norm = nn.Identity()
        nn.init.zeros_(self.attn.wo.weight)

    def forward(self, features, rois, tokens, stage):
        bsz, nroi, _ = rois.shape
        x = features.reshape(bsz, nroi, -1)
        low, high = map(math.radians, self.encoder.rv_elevation_bounds)
        result = []
        active = 0
        for b, (rv, uv) in enumerate(tokens):
            box = rois[b].detach()
            keep = (box[:, 3:6] > 0).all(dim=1)
            if not keep.any() or not len(rv):
                result.append(x[b])
                continue
            xyz = box[keep, :3]
            az = torch.atan2(xyz[:, 1], xyz[:, 0])
            el = torch.asin((xyz[:, 2]/xyz.norm(dim=1).clamp_min(1e-6)).clamp(-1, 1))
            query_uv = torch.stack(((az+math.pi)/(2*math.pi), (el-low)/(high-low)), dim=1)
            updated = x[b].clone()
            updated[keep] = self.attn(x[b, keep], query_uv, rv, uv)
            result.append(updated)
            active += 1
        pred = torch.stack(result).reshape_as(features)
        if not active:
            # Keep DDP collectives consistent for a completely empty frame/proposal batch.
            pred = pred + sum(p.reshape(-1)[0] * 0. for p in self.parameters())
        return pred


def make_geo(config, channels):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(config.INIT_SEED)
        return CascadeRVClassification(config, channels)
