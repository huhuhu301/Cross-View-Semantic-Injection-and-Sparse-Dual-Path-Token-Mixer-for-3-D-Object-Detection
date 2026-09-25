# Modified by RV-SDTM contributors for this public release; see NOTICE.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

from .spconv_backbone import post_act_block, SparseBasicBlock
from pcdet.models.model_utils.sparse_block_utils import (
    post_act_block_sparse_3d, post_act_block_sparse_2d,
    SparseBasicBlock3D, SparseBasicBlock2D
)
from ...utils.spconv_utils import replace_feature, spconv
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils.loss_utils import focal_loss_sparse
from pcdet.models.model_utils.cross_view_dca import build_cross_view_fusion, fuse_rv_features


norm_fn_1d = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)


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

    def _compute_scores(self, feats):
        strategy = self.selection_strategy
        if strategy in ['importance', 'hybrid']:
            return feats.pow(2).sum(dim=1)
        if strategy in ['learned', 'learned_hybrid'] and self.score_head is not None:
            scores = self.score_head(feats).squeeze(-1)
            if self.training and self.score_noise > 0:
                scores = scores + torch.randn_like(scores) * self.score_noise
            if self.score_temperature != 1.0:
                scores = scores / self.score_temperature
            return scores
        return feats.new_ones(feats.shape[0])

    def _random_select(self, candidate_idx, num):
        if num <= 0 or candidate_idx.numel() == 0:
            return candidate_idx.new_empty(0, dtype=torch.long)
        if candidate_idx.numel() <= num:
            return candidate_idx
        perm = torch.randperm(candidate_idx.shape[0], device=candidate_idx.device)
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


class AttnPillarPool(nn.Module):
    def __init__(self, dim, pillar_size=6):
        super().__init__()
        self.dim = dim
        self.pillar_size = pillar_size
        self.query_func = spconv.SparseMaxPool3d((pillar_size, 1, 1), stride=(pillar_size, 1, 1), padding=0)
        self.norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.pos_embedding = nn.Embedding(pillar_size, dim)
        nn.init.normal_(self.pos_embedding.weight, std=.01)

    def forward(self, x):
        src = self.query_func(x)
        _, batch_win_inds = torch.unique(x.indices[:, [0, 2, 3]], return_inverse=True, dim=0)
        num_pillars = int(batch_win_inds.max() + 1)
        scatter_indices = torch.stack([batch_win_inds, x.indices[:, 1]], -1)

        def _scatter_nd(indices: torch.Tensor, updates: torch.Tensor, shape: torch.Size):
            out = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
            out[indices[:, 0], indices[:, 1]] = updates
            return out

        key = value = _scatter_nd(
            scatter_indices, x.features,
            torch.Size([num_pillars, self.pillar_size, self.dim])
        )
        key_padding_mask = ~_scatter_nd(
            scatter_indices,
            torch.ones_like(x.features[:, 0]),
            torch.Size([num_pillars, self.pillar_size])
        ).bool()
        key = key + self.pos_embedding.weight.unsqueeze(0).repeat(num_pillars, 1, 1)
        # Pillar attention and its residual normalization accumulate in FP32.
        with torch.cuda.amp.autocast(enabled=False):
            query_fp32 = src.features.float().unsqueeze(1)
            key_fp32 = key.float()
            value_fp32 = value.float()
            out = self.self_attn(
                query_fp32, key_fp32, value_fp32, key_padding_mask
            )[0].squeeze(1)
            out = self.norm(out + src.features.float())
        src = replace_feature(src, out)
        return src


class SEDLayer(spconv.SparseModule):

    def __init__(self, dim: int, down_kernel_size: list, down_stride: list, num_SBB: list, indice_key, xy_only=False, bias=False):
        super().__init__()
        block = SparseBasicBlock2D if xy_only else SparseBasicBlock3D
        post_act = post_act_block_sparse_2d if xy_only else post_act_block_sparse_3d

        self.encoder = nn.ModuleList([spconv.SparseSequential(*[block(dim, indice_key=f"{indice_key}_0") for _ in range(num_SBB[0])])])
        num_lv = len(down_stride)
        for i in range(1, num_lv):
            cur = [
                post_act(dim, dim, down_kernel_size[i], down_stride[i], down_kernel_size[i] // 2,
                         conv_type='spconv', indice_key=f'spconv_{indice_key}_{i}'),
                *[block(dim, indice_key=f"{indice_key}_{i}", bias=bias) for _ in range(num_SBB[i])]
            ]
            self.encoder.append(spconv.SparseSequential(*cur))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for i in range(num_lv - 1, 0, -1):
            self.decoder.append(post_act(dim, dim, down_kernel_size[i], conv_type='inverseconv', indice_key=f'spconv_{indice_key}_{i}'))
            self.decoder_norm.append(norm_fn_1d(dim))

    def forward(self, x):
        feats = []
        for conv in self.encoder:
            x = conv(x)
            feats.append(x)
        x = feats[-1]
        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, feats[:-1][::-1]):
            x = deconv(x)
            x = replace_feature(x, norm(x.features + up_x.features))
        return x


class RVSDTM(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, class_names, voxel_size, point_cloud_range, **kwargs):
        super().__init__()

        self.input_channels = input_channels
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        dim = model_cfg.FEATURE_DIM
        self.cpe = spconv.SparseSequential(
            post_act_block(input_channels, dim, 3, norm_fn=norm_fn, padding=1, indice_key='subm1', conv_type='subm'),
            SparseBasicBlock(dim, dim, norm_fn=norm_fn, indice_key='stem'),
            SparseBasicBlock(dim, dim, norm_fn=norm_fn, indice_key='stem'),
            SparseBasicBlock(dim, dim, norm_fn=norm_fn, indice_key='stem'),
            post_act_block(dim, dim, (3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=0, indice_key='spconv1', conv_type='spconv'),
        )

        sdtm_cfg = model_cfg.get('SDTM', {})
        router_dim = sdtm_cfg.get('ROUTER_DIM', dim)
        router_stride = sdtm_cfg.get('ROUTER_STRIDE', 4)
        router_heads = sdtm_cfg.get('ROUTER_HEADS', 4)
        router_attn_dim = sdtm_cfg.get('ROUTER_ATTN_DIM', router_dim)
        router_keep_ratio = sdtm_cfg.get('ROUTER_KEEP_RATIO', 1.0)
        router_selection = sdtm_cfg.get('ROUTER_SELECTION', 'importance')
        router_selection_cfg = sdtm_cfg.get('ROUTER_SELECTION_CFG', {})
        blocks_stage1 = sdtm_cfg.get('NUM_BLOCKS_STAGE1', 2)
        blocks_stage2 = sdtm_cfg.get('NUM_BLOCKS_STAGE2', 2)
        use_router_stage1 = sdtm_cfg.get('USE_ROUTER_STAGE1', True)
        use_router_stage2 = sdtm_cfg.get('USE_ROUTER_STAGE2', True)

        def make_stage(num_blocks, tag, use_router):
            layers = []
            for i in range(num_blocks):
                layers.append(
                    SDTMBlock(
                        dim=dim,
                        router_dim=router_dim,
                        stride_xy=router_stride,
                        attn_heads=router_heads,
                        attn_dim=router_attn_dim,
                        indice_tag=f'{tag}_{i}',
                        use_router=use_router,
                        keep_ratio=router_keep_ratio,
                        selection_strategy=router_selection,
                        selection_cfg=router_selection_cfg,
                        use_local=sdtm_cfg.get('USE_LOCAL', True),
                        use_gate=sdtm_cfg.get('USE_GATE', True),
                    )
                )
            return spconv.SparseSequential(*layers)

        self.stage1 = spconv.SparseSequential(
            make_stage(blocks_stage1, 'stage1', use_router_stage1),
            nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
            spconv.SparseConv3d(dim, dim, 3, stride=2, padding=1, bias=False, indice_key='down_sdtm'),
        )
        self.stage2 = make_stage(blocks_stage2, 'stage2', use_router_stage2)

        pillar_size = int(self.sparse_shape[0] // 4)
        pool_mode = model_cfg.get('APPOOL_MODE', 'attention')
        if pool_mode not in ('attention', 'max'):
            raise ValueError('APPOOL_MODE must be attention or max')
        self.app = (AttnPillarPool(dim, pillar_size) if pool_mode == 'attention' else
                    spconv.SparseMaxPool3d((pillar_size, 1, 1), stride=(pillar_size, 1, 1)))

        self.rv_projection, self.cross_view_dca = build_cross_view_fusion(model_cfg, input_channels)
        self.use_rv = self.cross_view_dca is not None

        self.adaptive_feature_diffusion = model_cfg.get('AFD', False)
        if self.adaptive_feature_diffusion:
            self.class_names = class_names
            self.voxel_size = voxel_size
            self.point_cloud_range = point_cloud_range
            self.fg_thr = model_cfg['FG_THRESHOLD']
            self.featmap_stride = model_cfg['FEATMAP_STRIDE']
            self.group_pooling_kernel_size = model_cfg.get(
                'GROUP_POOLING_KERNEL_SIZE',
                model_cfg.get('GREOUP_POOLING_KERNEL_SIZE'),
            )
            if self.group_pooling_kernel_size is None:
                raise KeyError(
                    'AFD requires BACKBONE_3D.GROUP_POOLING_KERNEL_SIZE'
                )
            self.detach_feature = model_cfg['DETACH_FEATURE']

            self.group_class_names = []
            for names in model_cfg['GROUP_CLASS_NAMES']:
                self.group_class_names.append([x for x in names if x in class_names])

            self.cls_conv = spconv.SparseSequential(
                spconv.SubMConv2d(dim, dim, 3, stride=1, padding=1, bias=False, indice_key='conv_cls'),
                norm_fn(dim),
                nn.ReLU(),
                spconv.SubMConv2d(dim, len(self.group_class_names), 1, bias=True, indice_key='cls_out')
            )
            self.forward_ret_dict = {}

        afd_dim = model_cfg.AFD_FEATURE_DIM
        afd_num_layers = model_cfg.AFD_NUM_LAYERS
        afd_num_SBB = model_cfg.AFD_NUM_SBB
        afd_down_kernel_size = model_cfg.AFD_DOWN_KERNEL_SIZE
        afd_down_stride = model_cfg.AFD_DOWN_STRIDE
        assert afd_down_stride[0] == 1
        assert len(afd_num_SBB) == len(afd_down_stride)

        self.bev_proj = spconv.SparseSequential(
            spconv.SubMConv2d(dim, afd_dim, 1, stride=1, padding=0, bias=False),
            norm_fn_1d(afd_dim),
            nn.ReLU()
        )

        self.afd_layers = nn.ModuleList()
        for idx in range(afd_num_layers):
            layer = SEDLayer(
                afd_dim, afd_down_kernel_size, afd_down_stride, afd_num_SBB,
                indice_key=f'afdlayer{idx}', xy_only=True
            )
            self.afd_layers.append(layer)

        self.up_block = spconv.SparseSequential(
            spconv.SparseConv2d(afd_dim, afd_dim, 3, stride=1, padding=1, bias=False, indice_key='up_conv'),
            norm_fn_1d(afd_dim),
            nn.ReLU()
        )
        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(afd_dim, afd_dim, 3, stride=1, padding=1, bias=False),
            norm_fn_1d(afd_dim),
            nn.ReLU()
        )

        self.num_point_features = dim
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (spconv.SubMConv2d, spconv.SubMConv3d, spconv.SparseConv2d, spconv.SparseConv3d)):
                nn.init.kaiming_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if self.adaptive_feature_diffusion:
            self.cls_conv[-1].bias.data.fill_(-2.19)

    def assign_target(self, batch_spatial_indices, batch_gt_boxes):
        all_names = np.array(['bg', *self.class_names])
        inside_box_target = batch_spatial_indices.new_zeros((len(self.group_class_names), batch_spatial_indices.shape[0]))

        for gidx, names in enumerate(self.group_class_names):
            batch_inside_box_mask = []
            for bidx in range(len(batch_gt_boxes)):
                spatial_indices = batch_spatial_indices[batch_spatial_indices[:, 0] == bidx][:, [2, 1]]
                points = spatial_indices.clone() + 0.5
                points[:, 0] = points[:, 0] * self.featmap_stride * self.voxel_size[0] + self.point_cloud_range[0]
                points[:, 1] = points[:, 1] * self.featmap_stride * self.voxel_size[1] + self.point_cloud_range[1]
                points = torch.cat([points, points.new_zeros((points.shape[0], 1))], dim=-1)

                gt_boxes = batch_gt_boxes[bidx].clone()
                gt_boxes = gt_boxes[(gt_boxes[:, 3] > 0) & (gt_boxes[:, 4] > 0)]
                gt_class_names = all_names[gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = [gt_boxes[_] for _, name in enumerate(gt_class_names) if name in names]
                inside_box_mask = points.new_zeros((points.shape[0]))
                if len(gt_boxes_single_head) > 0:
                    boxes = torch.stack(gt_boxes_single_head)[:, :7]
                    boxes[:, 2] = 0
                    inside_box_mask[roiaware_pool3d_utils.points_in_boxes_gpu(points[None], boxes[None])[0] > -1] = 1
                batch_inside_box_mask.append(inside_box_mask)
            inside_box_target[gidx] = torch.cat(batch_inside_box_mask)
        return inside_box_target

    def get_loss(self):
        spatial_indices = self.forward_ret_dict['spatial_indices']
        batch_size = self.forward_ret_dict['batch_size']
        batch_index = spatial_indices[:, 0]

        inside_box_pred = self.forward_ret_dict['inside_box_pred']
        inside_box_target = self.forward_ret_dict['inside_box_target']
        inside_box_pred = torch.cat([inside_box_pred[:, batch_index == bidx] for bidx in range(batch_size)], dim=1)
        inside_box_pred = torch.clamp(inside_box_pred.sigmoid(), min=1e-4, max=1 - 1e-4)

        cls_loss = 0.0
        recall_dict = {}
        for gidx in range(len(self.group_class_names)):
            group_cls_loss = focal_loss_sparse(inside_box_pred[gidx], inside_box_target[gidx].float())
            cls_loss += group_cls_loss

            fg_mask = inside_box_target[gidx] > 0
            pred_mask = inside_box_pred[gidx][fg_mask] > self.fg_thr
            recall_dict[f'afd_recall_{gidx}'] = (pred_mask.sum() / fg_mask.sum().clamp(min=1.0)).item()
            recall_dict[f'afd_cls_loss_{gidx}'] = group_cls_loss.item()

        return cls_loss, recall_dict

    def feature_diffusion(self, x, batch_dict):
        if not self.adaptive_feature_diffusion:
            return x

        detached_x = x
        if self.detach_feature:
            detached_x = spconv.SparseConvTensor(
                features=x.features.detach(),
                indices=x.indices,
                spatial_shape=x.spatial_shape,
                batch_size=x.batch_size
            )

        inside_box_pred = self.cls_conv(detached_x).features.permute(1, 0)

        if self.training:
            inside_box_target = self.assign_target(x.indices, batch_dict['gt_boxes'])
            self.forward_ret_dict['batch_size'] = x.batch_size
            self.forward_ret_dict['spatial_indices'] = x.indices
            self.forward_ret_dict['inside_box_pred'] = inside_box_pred
            self.forward_ret_dict['inside_box_target'] = inside_box_target

        group_inside_mask = inside_box_pred.sigmoid() > self.fg_thr
        bg_mask = ~group_inside_mask.max(dim=0, keepdim=True)[0]
        group_inside_mask = torch.cat([group_inside_mask, bg_mask], dim=0)

        one_mask = x.features.new_zeros((x.batch_size, 1, x.spatial_shape[0], x.spatial_shape[1]))
        for gidx, inside_mask in enumerate(group_inside_mask):
            selected_indices = x.indices[inside_mask]
            single_one_mask = spconv.SparseConvTensor(
                features=x.features.new_ones(selected_indices.shape[0], 1),
                indices=selected_indices,
                spatial_shape=x.spatial_shape,
                batch_size=x.batch_size
            ).dense()
            pooling_size = self.group_pooling_kernel_size[gidx]
            single_one_mask = F.max_pool2d(single_one_mask, kernel_size=pooling_size, stride=1, padding=pooling_size // 2)
            one_mask = torch.maximum(one_mask, single_one_mask)

        zero_indices = (one_mask[:, 0] > 0).nonzero().int()
        zero_features = x.features.new_zeros((len(zero_indices), x.features.shape[1]))

        cat_indices = torch.cat([x.indices, zero_indices], dim=0)
        cat_features = torch.cat([x.features, zero_features], dim=0)
        indices_unique, _inv = torch.unique(cat_indices, dim=0, return_inverse=True)
        features_unique = x.features.new_zeros((indices_unique.shape[0], x.features.shape[1]))
        features_unique.index_add_(0, _inv, cat_features)

        x = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size
        )
        return x

    def rv_fusion(self, x, batch_dict):
        features = fuse_rv_features(x.features, x.indices[:, 0], batch_dict,
                                    self.rv_projection, self.cross_view_dca)
        return replace_feature(x, features)

    @staticmethod
    def to_bev(x):
        features = x.features
        indices = x.indices[:, [0, 2, 3]]
        spatial_shape = x.spatial_shape[1:]

        x = spconv.SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=spatial_shape,
            batch_size=x.batch_size
        )
        return x

    def upsampling(self, x, up_stride=2, is_diffusion=True):
        x.indices[:, 1:] *= up_stride
        x.spatial_shape = [s * up_stride for s in x.spatial_shape]

        if is_diffusion and x.indices.shape[0] > 0:
            x = self.up_block(x)

        return x

    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.rv_fusion(x, batch_dict)
        x = self.cpe(x)
        x = self.stage1(x)
        x = self.stage2(x)

        x = self.app(x)
        x = self.to_bev(x)

        x = self.feature_diffusion(x, batch_dict)
        x = self.bev_proj(x)
        for layer in self.afd_layers:
            x = layer(x)

        x = self.upsampling(x)
        x = self.shared_conv(x)

        batch_dict.update({
            'spatial_features_2d': x,
        })
        return batch_dict
