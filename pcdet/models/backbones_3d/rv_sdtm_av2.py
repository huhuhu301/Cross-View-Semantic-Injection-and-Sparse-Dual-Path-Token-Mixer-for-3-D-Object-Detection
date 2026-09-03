# Modified by RV-SDTM contributors for this public release; see NOTICE.
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.models.model_utils.cross_view_dca import CrossViewDCA
from pcdet.models.model_utils.sparse_block_utils import (
    SparseBasicBlock3D,
    post_act_block_sparse_3d,
)

from .rv_sdtm import AttnPillarPool, SDTMBlock, SEDLayer
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils.loss_utils import focal_loss_sparse
from ...utils.spconv_utils import replace_feature, spconv


norm_fn_1d = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)


class RVSDTMAV2(nn.Module):
    """Argoverse2 geometry adapter for RV-SDTM.

    AV2 uses a 400 m by 400 m XY range with 0.1 m voxels. Reusing the Waymo
    RV-SDTM path would restore stride-1 BEV features and make the sparse head
    operate on a 4000 by 4000 grid. This adapter retains the AV2 sparse output
    contract instead:

        input -> stride-4 SDTM stage -> stride-8 SDTM stage -> APPool/AFD
              -> 2x sparse upsampling -> stride-4 SparseDynamicHead features.

    RV features are injected before any sparse downsampling, so voxel/RV view
    anchors still have a one-to-one correspondence at the GeoCVA call site.
    """

    def __init__(
            self, model_cfg, input_channels, grid_size, class_names,
            voxel_size, point_cloud_range, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.input_channels = input_channels
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        dim = model_cfg.FEATURE_DIM
        self.feature_dim = dim

        # Progressive sparse stem for the long-range AV2 path.
        # Two 3-D stride-2 layers put SDTM stage 1 at XY stride 4 and reduce
        # the 41-bin padded Z shape to 11 bins.
        self.stem = spconv.SparseSequential(
            post_act_block_sparse_3d(
                input_channels, 16, 3, 1, 1,
                indice_key='argo2_sdtm_subm1', conv_type='subm'),
            SparseBasicBlock3D(16, indice_key='argo2_sdtm_conv1'),
            SparseBasicBlock3D(16, indice_key='argo2_sdtm_conv1'),
            post_act_block_sparse_3d(
                16, 32, 3, 2, 1,
                indice_key='argo2_sdtm_spconv1', conv_type='spconv'),
            SparseBasicBlock3D(32, indice_key='argo2_sdtm_conv2'),
            SparseBasicBlock3D(32, indice_key='argo2_sdtm_conv2'),
            post_act_block_sparse_3d(
                32, dim, 3, 2, 1,
                indice_key='argo2_sdtm_spconv2', conv_type='spconv'),
            SparseBasicBlock3D(dim, indice_key='argo2_sdtm_cpe'),
            SparseBasicBlock3D(dim, indice_key='argo2_sdtm_cpe'),
            SparseBasicBlock3D(dim, indice_key='argo2_sdtm_cpe'),
        )

        sdtm_cfg = model_cfg.get('SDTM', {})
        router_dim = sdtm_cfg.get('ROUTER_DIM', dim)
        router_stride = sdtm_cfg.get('ROUTER_STRIDE', 4)
        router_heads = sdtm_cfg.get('ROUTER_HEADS', 4)
        router_attn_dim = sdtm_cfg.get('ROUTER_ATTN_DIM', router_dim)
        router_keep_ratio = sdtm_cfg.get('ROUTER_KEEP_RATIO', 1.0)
        router_selection = sdtm_cfg.get('ROUTER_SELECTION', 'importance')
        router_selection_cfg = sdtm_cfg.get('ROUTER_SELECTION_CFG', {})
        blocks_stage1 = sdtm_cfg.get('NUM_BLOCKS_STAGE1', 1)
        blocks_stage2 = sdtm_cfg.get('NUM_BLOCKS_STAGE2', 1)
        use_router_stage1 = sdtm_cfg.get('USE_ROUTER_STAGE1', False)
        use_router_stage2 = sdtm_cfg.get('USE_ROUTER_STAGE2', True)

        def make_stage(num_blocks, tag, use_router):
            return spconv.SparseSequential(*[
                SDTMBlock(
                    dim=dim,
                    router_dim=router_dim,
                    stride_xy=router_stride,
                    attn_heads=router_heads,
                    attn_dim=router_attn_dim,
                    indice_tag=f'argo2_{tag}_{idx}',
                    use_router=use_router,
                    keep_ratio=router_keep_ratio,
                    selection_strategy=router_selection,
                    selection_cfg=router_selection_cfg,
                )
                for idx in range(num_blocks)
            ])

        self.stage1 = make_stage(blocks_stage1, 'stage1', use_router_stage1)
        self.stage_down = spconv.SparseSequential(
            norm_fn_1d(dim),
            spconv.SparseConv3d(
                dim, dim, 3, stride=(1, 2, 2), padding=1, bias=False,
                indice_key='argo2_sdtm_stage_down'),
        )
        self.stage2 = make_stage(blocks_stage2, 'stage2', use_router_stage2)

        z_after_stem = math.ceil(math.ceil(int(self.sparse_shape[0]) / 2) / 2)
        configured_pillar_size = model_cfg.get('PILLAR_SIZE', z_after_stem)
        if int(configured_pillar_size) != z_after_stem:
            raise ValueError(
                f'BACKBONE_3D.PILLAR_SIZE={configured_pillar_size} does not '
                f'match the AV2 stem Z shape {z_after_stem}'
            )
        self.pillar_size = z_after_stem
        self.app = AttnPillarPool(dim, self.pillar_size)

        self.use_rv = model_cfg.get('USE_RV', False)
        if self.use_rv:
            rv_feature_dim = model_cfg.get('RV_FEATURE_DIM', None)
            if rv_feature_dim is None:
                raise ValueError(
                    'USE_RV=True requires BACKBONE_3D.RV_FEATURE_DIM to '
                    'match VFE.RV_FEATURE_DIM'
                )
            rv_attn_dim = model_cfg.get('RV_ATTN_DIM', dim)
            self.rv_projection = (
                nn.Linear(rv_feature_dim, rv_attn_dim, bias=False)
                if rv_feature_dim != rv_attn_dim else nn.Identity()
            )
            self.cross_view_dca = CrossViewDCA(
                vox_c=input_channels,
                rv_c=rv_attn_dim,
                attn_dim=rv_attn_dim,
                heads=model_cfg.get('RV_HEADS', 8),
                k=model_cfg.get('RV_DCA_K', 6),
                max_rv_tokens=model_cfg.get('RV_MAX_TOKENS', 12000),
                vox_chunk=model_cfg.get('RV_VOX_CHUNK', 1024),
                su=model_cfg.get('RV_SU', 0.05),
                sv=model_cfg.get('RV_SV', 0.08),
                sigma=model_cfg.get('RV_SIGMA', 1.0),
                tau=model_cfg.get('RV_TAU', 1.0),
                alpha_init=model_cfg.get('RV_ALPHA_INIT', 0.3),
            )

        self.adaptive_feature_diffusion = model_cfg.get('AFD', False)
        if self.adaptive_feature_diffusion:
            self.class_names = class_names
            self.fg_thr = model_cfg.FG_THRESHOLD
            self.featmap_stride = model_cfg.FEATMAP_STRIDE
            self.group_pooling_kernel_size = model_cfg.get(
                'GROUP_POOLING_KERNEL_SIZE',
                model_cfg.get('GREOUP_POOLING_KERNEL_SIZE'),
            )
            if self.group_pooling_kernel_size is None:
                raise KeyError(
                    'AFD requires BACKBONE_3D.GROUP_POOLING_KERNEL_SIZE'
                )
            self.detach_feature = model_cfg.DETACH_FEATURE
            self.group_class_names = [
                [name for name in names if name in class_names]
                for names in model_cfg.GROUP_CLASS_NAMES
            ]
            self.cls_conv = spconv.SparseSequential(
                spconv.SubMConv2d(
                    dim, dim, 3, stride=1, padding=1, bias=False,
                    indice_key='argo2_sdtm_conv_cls'),
                norm_fn_1d(dim),
                nn.ReLU(),
                spconv.SubMConv2d(
                    dim, len(self.group_class_names), 1, bias=True,
                    indice_key='argo2_sdtm_cls_out'),
            )
            self.forward_ret_dict = {}

        afd_dim = model_cfg.AFD_FEATURE_DIM
        afd_num_layers = model_cfg.AFD_NUM_LAYERS
        afd_num_sbb = model_cfg.AFD_NUM_SBB
        afd_down_kernel_size = model_cfg.AFD_DOWN_KERNEL_SIZE
        afd_down_stride = model_cfg.AFD_DOWN_STRIDE
        if afd_down_stride[0] != 1:
            raise ValueError('AFD_DOWN_STRIDE must start at 1')
        if not (
                len(afd_num_sbb) == len(afd_down_kernel_size)
                == len(afd_down_stride)):
            raise ValueError('AFD SBB, kernel, and stride lists must align')

        self.bev_proj = spconv.SparseSequential(
            spconv.SubMConv2d(dim, afd_dim, 1, bias=False),
            norm_fn_1d(afd_dim),
            nn.ReLU(),
        )
        self.afd_layers = nn.ModuleList([
            SEDLayer(
                afd_dim, afd_down_kernel_size, afd_down_stride, afd_num_sbb,
                indice_key=f'argo2_sdtm_afd_{idx}', xy_only=True)
            for idx in range(afd_num_layers)
        ])
        self.up_block = spconv.SparseSequential(
            spconv.SparseConv2d(
                afd_dim, afd_dim, 3, stride=1, padding=1, bias=False,
                indice_key='argo2_sdtm_up_conv'),
            norm_fn_1d(afd_dim),
            nn.ReLU(),
        )
        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(
                afd_dim, afd_dim, 3, stride=1, padding=1, bias=False,
                indice_key='argo2_sdtm_shared_conv'),
            norm_fn_1d(afd_dim),
            nn.ReLU(),
        )

        self.num_point_features = afd_dim
        self.init_weights()

    def init_weights(self):
        sparse_convs = (
            spconv.SubMConv2d,
            spconv.SubMConv3d,
            spconv.SparseConv2d,
            spconv.SparseConv3d,
        )
        for module in self.modules():
            if isinstance(module, sparse_convs):
                nn.init.kaiming_normal_(module.weight)
                if getattr(module, 'bias', None) is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(module.weight, 1)
                if getattr(module, 'bias', None) is not None:
                    nn.init.constant_(module.bias, 0)
        if self.adaptive_feature_diffusion:
            self.cls_conv[-1].bias.data.fill_(-2.19)

    def rv_fusion(self, x, batch_dict):
        if not self.use_rv:
            return x

        required_keys = ('rv_tokens', 'rv_uv', 'vox_uv')
        missing = [key for key in required_keys if key not in batch_dict]
        if missing:
            raise KeyError(f'RangeViewVFE did not provide: {missing}')

        vox_uv = batch_dict['vox_uv']
        if vox_uv.shape[0] != x.features.shape[0]:
            raise ValueError(
                'GeoCVA requires one view anchor per input voxel: '
                f'{vox_uv.shape[0]} anchors vs {x.features.shape[0]} voxels'
            )

        rv_tokens = [self.rv_projection(tokens) for tokens in batch_dict['rv_tokens']]
        rv_uv = batch_dict['rv_uv']
        features = x.features.clone()
        for batch_idx in range(x.batch_size):
            voxel_mask = x.indices[:, 0] == batch_idx
            if voxel_mask.any() and rv_tokens[batch_idx].numel() > 0:
                features[voxel_mask] = self.cross_view_dca(
                    features[voxel_mask],
                    vox_uv[voxel_mask],
                    rv_tokens[batch_idx],
                    rv_uv[batch_idx],
                )
        return replace_feature(x, features)

    def assign_target(self, batch_spatial_indices, batch_gt_boxes):
        all_names = np.array(['bg', *self.class_names])
        target = batch_spatial_indices.new_zeros(
            (len(self.group_class_names), batch_spatial_indices.shape[0]))

        for group_idx, names in enumerate(self.group_class_names):
            batch_targets = []
            for batch_idx in range(len(batch_gt_boxes)):
                spatial_indices = batch_spatial_indices[
                    batch_spatial_indices[:, 0] == batch_idx][:, [2, 1]]
                points = spatial_indices.clone() + 0.5
                points[:, 0] = (
                    points[:, 0] * self.featmap_stride * self.voxel_size[0]
                    + self.point_cloud_range[0]
                )
                points[:, 1] = (
                    points[:, 1] * self.featmap_stride * self.voxel_size[1]
                    + self.point_cloud_range[1]
                )
                points = torch.cat(
                    [points, points.new_zeros((points.shape[0], 1))], dim=-1)

                gt_boxes = batch_gt_boxes[batch_idx].clone()
                gt_boxes = gt_boxes[
                    (gt_boxes[:, 3] > 0) & (gt_boxes[:, 4] > 0)]
                gt_class_names = all_names[
                    gt_boxes[:, -1].cpu().long().numpy()]
                group_boxes = [
                    gt_boxes[idx] for idx, name in enumerate(gt_class_names)
                    if name in names
                ]
                inside_mask = points.new_zeros(points.shape[0])
                if group_boxes:
                    boxes = torch.stack(group_boxes)[:, :7]
                    boxes[:, 2] = 0
                    inside_mask[
                        roiaware_pool3d_utils.points_in_boxes_gpu(
                            points[None], boxes[None])[0] > -1
                    ] = 1
                batch_targets.append(inside_mask)
            target[group_idx] = torch.cat(batch_targets)
        return target

    def get_loss(self):
        spatial_indices = self.forward_ret_dict['spatial_indices']
        batch_size = self.forward_ret_dict['batch_size']
        batch_index = spatial_indices[:, 0]
        inside_box_pred = self.forward_ret_dict['inside_box_pred']
        inside_box_target = self.forward_ret_dict['inside_box_target']
        inside_box_pred = torch.cat([
            inside_box_pred[:, batch_index == batch_idx]
            for batch_idx in range(batch_size)
        ], dim=1)
        inside_box_pred = torch.clamp(
            inside_box_pred.sigmoid(), min=1e-4, max=1 - 1e-4)

        cls_loss = inside_box_pred.sum() * 0.0
        recall_dict = {}
        for group_idx in range(len(self.group_class_names)):
            group_loss = focal_loss_sparse(
                inside_box_pred[group_idx],
                inside_box_target[group_idx].float(),
            )
            cls_loss += group_loss
            foreground = inside_box_target[group_idx] > 0
            predicted = inside_box_pred[group_idx][foreground] > self.fg_thr
            recall_dict[f'afd_recall_{group_idx}'] = (
                predicted.sum() / foreground.sum().clamp(min=1.0)).item()
            recall_dict[f'afd_cls_loss_{group_idx}'] = group_loss.item()
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
                batch_size=x.batch_size,
            )

        inside_box_pred = self.cls_conv(detached_x).features.permute(1, 0)
        if self.training:
            inside_box_target = self.assign_target(x.indices, batch_dict['gt_boxes'])
            self.forward_ret_dict.update({
                'batch_size': x.batch_size,
                'spatial_indices': x.indices,
                'inside_box_pred': inside_box_pred,
                'inside_box_target': inside_box_target,
            })

        group_inside_mask = inside_box_pred.sigmoid() > self.fg_thr
        background_mask = ~group_inside_mask.max(dim=0, keepdim=True)[0]
        group_inside_mask = torch.cat(
            [group_inside_mask, background_mask], dim=0)

        one_mask = x.features.new_zeros(
            (x.batch_size, 1, x.spatial_shape[0], x.spatial_shape[1]))
        for group_idx, inside_mask in enumerate(group_inside_mask):
            selected_indices = x.indices[inside_mask]
            single_mask = spconv.SparseConvTensor(
                features=x.features.new_ones((selected_indices.shape[0], 1)),
                indices=selected_indices,
                spatial_shape=x.spatial_shape,
                batch_size=x.batch_size,
            ).dense()
            pooling_size = self.group_pooling_kernel_size[group_idx]
            single_mask = F.max_pool2d(
                single_mask,
                kernel_size=pooling_size,
                stride=1,
                padding=pooling_size // 2,
            )
            one_mask = torch.maximum(one_mask, single_mask)

        new_indices = (one_mask[:, 0] > 0).nonzero().int()
        new_features = x.features.new_zeros(
            (len(new_indices), x.features.shape[1]))
        cat_indices = torch.cat([x.indices, new_indices], dim=0)
        cat_features = torch.cat([x.features, new_features], dim=0)
        unique_indices, inverse = torch.unique(
            cat_indices, dim=0, return_inverse=True)
        unique_features = x.features.new_zeros(
            (unique_indices.shape[0], x.features.shape[1]))
        unique_features.index_add_(0, inverse, cat_features)
        return spconv.SparseConvTensor(
            features=unique_features,
            indices=unique_indices,
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size,
        )

    @staticmethod
    def to_bev(x):
        return spconv.SparseConvTensor(
            features=x.features,
            indices=x.indices[:, [0, 2, 3]],
            spatial_shape=x.spatial_shape[1:],
            batch_size=x.batch_size,
        )

    def upsampling(self, x, up_stride=2):
        x.indices[:, 1:] *= up_stride
        x.spatial_shape = [size * up_stride for size in x.spatial_shape]
        if x.indices.shape[0] > 0:
            x = self.up_block(x)
        return x

    def forward(self, batch_dict):
        x = spconv.SparseConvTensor(
            features=batch_dict['voxel_features'],
            indices=batch_dict['voxel_coords'].int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_dict['batch_size'],
        )

        x = self.rv_fusion(x, batch_dict)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage_down(x)
        x = self.stage2(x)

        if int(x.spatial_shape[0]) != self.pillar_size:
            raise RuntimeError(
                f'Unexpected AV2 Z shape {x.spatial_shape[0]}; '
                f'APPool expects {self.pillar_size}'
            )
        x = self.app(x)
        x = self.to_bev(x)
        x = self.feature_diffusion(x, batch_dict)
        x = self.bev_proj(x)
        for layer in self.afd_layers:
            x = layer(x)
        x = self.upsampling(x, up_stride=2)
        x = self.shared_conv(x)

        batch_dict['spatial_features_2d'] = x
        return batch_dict
