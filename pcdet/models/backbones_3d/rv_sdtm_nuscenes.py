# Modified by RV-SDTM contributors for this public release; see NOTICE.
from ...utils.spconv_utils import spconv
from .rv_sdtm import RVSDTM


class RVSDTMNuScenes(RVSDTM):
    """nuScenes geometry adapter for the RV-SDTM backbone.

    The shared implementation provides RV-GVFE/GeoCVA fusion, SDTM stages,
    attentional pillar pooling, and AFD. nuScenes already reaches the
    detector's stride-2 BEV resolution after pillar pooling, so the Waymo
    backbone's unconditional 2x sparse upsampling must not be applied.
    """

    def __init__(self, model_cfg, input_channels, grid_size, class_names,
                 voxel_size, point_cloud_range, **kwargs):
        super().__init__(
            model_cfg=model_cfg,
            input_channels=input_channels,
            grid_size=grid_size,
            class_names=class_names,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            **kwargs,
        )
        self.model_cfg = model_cfg
        self.out_stride = int(model_cfg.get('FEATMAP_STRIDE', 2))
        self.num_bev_features = int(model_cfg.AFD_FEATURE_DIM)
        self.afd_loss_weight = float(model_cfg.get('AFD_LOSS_WEIGHT', 0.0))

    def _target_bev_hw(self):
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        vx, vy, _ = self.voxel_size
        grid_x = int(round((x_max - x_min) / vx))
        grid_y = int(round((y_max - y_min) / vy))
        return grid_y // self.out_stride, grid_x // self.out_stride

    def _match_target_bev(self, x):
        target_h, target_w = self._target_bev_hw()
        current_h, current_w = map(int, x.spatial_shape[-2:])
        if (current_h, current_w) == (target_h, target_w):
            return x

        can_upsample = (
            current_h < target_h
            and current_w < target_w
            and target_h % current_h == 0
            and target_w % current_w == 0
            and target_h // current_h == target_w // current_w
        )
        if can_upsample:
            x = self.upsampling(x, up_stride=target_h // current_h)
            current_h, current_w = map(int, x.spatial_shape[-2:])

        if (current_h, current_w) != (target_h, target_w):
            raise RuntimeError(
                'RV-SDTM produced an incompatible nuScenes BEV shape: '
                f'got {(current_h, current_w)}, expected {(target_h, target_w)} '
                f'at stride {self.out_stride}.'
            )
        return x

    def forward(self, batch_dict):
        x = spconv.SparseConvTensor(
            features=batch_dict['voxel_features'],
            indices=batch_dict['voxel_coords'].int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_dict['batch_size'],
        )

        # Paper protocol: a single early RV-to-voxel GeoCVA injection.
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

        x = self._match_target_bev(x)
        x = self.shared_conv(x)
        batch_dict['spatial_features_2d'] = x
        return batch_dict
