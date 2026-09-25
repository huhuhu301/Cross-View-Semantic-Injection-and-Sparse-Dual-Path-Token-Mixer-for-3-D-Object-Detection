# Modified by RV-SDTM contributors for this public release; see NOTICE.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate
from .dynamic_pillar_vfe import PFNLayerV2
from pcdet.models.backbones_3d.range_view_layers import DEDLayer
# from torch_scatter import scatter_mean
from torch_scatter import scatter_mean



class DynamicVoxelVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        # Non-persistent buffers follow model placement without changing the
        # historical checkpoint schema.
        self.register_buffer('grid_size', torch.as_tensor(grid_size), persistent=False)
        self.register_buffer('voxel_size', torch.as_tensor(voxel_size), persistent=False)
        self.register_buffer(
            'point_cloud_range', torch.as_tensor(point_cloud_range), persistent=False)

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor((points[:, [1,2,3]] - self.point_cloud_range[[0,1,2]]) / self.voxel_size[[0,1,2]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1,2]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xyz + \
                       points_coords[:, 0] * self.scale_yz + \
                       points_coords[:, 1] * self.scale_z + \
                       points_coords[:, 2]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        # f_center[:, 2] = points_xyz[:, 2] - self.z_offset
        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords

        return batch_dict


class PFNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, last_layer=False):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.last_layer = last_layer
        if not last_layer:
            self.reduce = nn.Sequential(
                nn.Linear(2 * output_dim, output_dim, bias=False),
                nn.BatchNorm1d(output_dim, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )

    def forward(self, inputs, unq_inv):
        x = self.linear(inputs)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        if self.last_layer:
            return x_max
        return self.reduce(torch.cat([x, x_max[unq_inv, :]], dim=1))


class RangeViewVFE(VFETemplate):

    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.rv_enabled = model_cfg.get('RV_ENABLED', True)
        rv_feature_dim = model_cfg.RV_FEATURE_DIM if self.rv_enabled else 0
        rv_layer_stride = model_cfg.RV_LAYER_STRIDE
        rv_layer_blocks = model_cfg.RV_LAYER_BLOCKS
        rv_num_layers = model_cfg.RV_NUM_LAYERS
        if self.rv_enabled:
            self.rv_input_roj = nn.Sequential(
                nn.Conv2d(6, rv_feature_dim, 3, 1, 1), nn.BatchNorm2d(rv_feature_dim), nn.ReLU())
            self.rv_backbone = nn.Sequential(*[
                DEDLayer(rv_feature_dim, rv_layer_stride, rv_layer_blocks) for _ in range(rv_num_layers)])

        self.rv_feature_fov = model_cfg.RV_FEATURE_FOV
        self.rv_feature_shape = model_cfg.RV_FEATURE_SHAPE
        self.num_filters = self.model_cfg.NUM_FILTERS
        num_point_features = num_point_features + 6 + rv_feature_dim
        num_filters = [num_point_features] + list(self.num_filters)
        self.pfn_layers = nn.ModuleList([])
        for i in range(len(num_filters) - 1):
            self.pfn_layers.append(
                PFNLayer(*num_filters[i: i + 2], last_layer=(i >= len(num_filters) - 2)))

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]
        # Geometry constants follow the module device.
        self.register_buffer(
            'grid_size', torch.as_tensor(grid_size), persistent=False)
        self.register_buffer(
            'voxel_size', torch.as_tensor(voxel_size), persistent=False)
        self.register_buffer(
            'point_cloud_range', torch.as_tensor(point_cloud_range),
            persistent=False)
        self.register_buffer(
            'offset', self.voxel_size[:3] / 2 + self.point_cloud_range[:3],
            persistent=False)

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points']  # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[None, :3]) / self.voxel_size[None, :3]).int()
        valid = ((points_coords >= 0) & (points_coords < self.grid_size[None, :3])).all(dim=1)
        points, points_coords = points[valid], points_coords[valid]
        if self.rv_enabled:
            rv_input, linear_indices, pixel_counts, u_float, v_float = self.create_range_view_feature(
                points, *self.rv_feature_shape, self.rv_feature_fov, batch_size=batch_dict['batch_size'])
            rv_features = self.rv_backbone(self.rv_input_roj(rv_input))
            B, C, H, W = rv_features.shape
            non_zero_mask = pixel_counts.reshape(B, H * W) > 0
            rv_tokens = rv_features.flatten(2).permute(0, 2, 1).contiguous()
            pixel_ranges = rv_input[:, 3].reshape(B, H * W)
            rv_tokens_list, rv_uv_list, rv_ranges_list = [], [], []
            u_grid = torch.arange(W, device=points.device).repeat(H)
            v_grid = torch.arange(H, device=points.device).repeat_interleave(W)
            for b in range(B):
                mask_b = non_zero_mask[b]
                rv_tokens_list.append(rv_tokens[b][mask_b])
                rv_ranges_list.append(pixel_ranges[b][mask_b])
                uv = torch.stack([u_grid[mask_b], v_grid[mask_b]], dim=1).float()
                rv_uv_list.append(uv / uv.new_tensor([W - 1, H - 1]))
            batch_dict['rv_tokens'] = rv_tokens_list
            batch_dict['rv_uv'] = rv_uv_list
            batch_dict['rv_ranges'] = rv_ranges_list
            point_rv = rv_features.permute(0, 2, 3, 1).reshape(-1, C)[linear_indices]

        merge_coords = (points[:, 0].long() * self.scale_xyz + points_coords[:, 0].long() * self.scale_yz
                        + points_coords[:, 1].long() * self.scale_z + points_coords[:, 2])
        unq_coords, unq_inv = torch.unique(merge_coords, return_inverse=True, dim=0)
        points_xyz = points[:, 1:4]
        points_mean = scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv]
        f_center = points_xyz - (points_coords.to(points.dtype) * self.voxel_size + self.offset)
        descriptor = [points[:, 1:], f_cluster, f_center]
        if self.rv_enabled:
            descriptor.append(point_rv)
            uv = torch.stack([u_float / (W - 1), v_float / (H - 1)], dim=1)
            batch_dict['vox_uv'] = scatter_mean(uv, unq_inv, dim=0)
            batch_dict['vox_ranges'] = scatter_mean(points_xyz.norm(dim=1), unq_inv, dim=0)
        if points.shape[0]:
            features = torch.cat(descriptor, dim=-1)
            for pfn in self.pfn_layers:
                features = pfn(features, unq_inv)
        else:
            features = points.new_zeros((0, self.num_filters[-1]))
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    unq_coords % self.scale_z,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    (unq_coords % self.scale_xyz) // self.scale_yz), dim=1).int()
        batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
        batch_dict['voxel_coords'] = voxel_coords
        return batch_dict


    def create_range_view_feature(self, points, H=128, W=1024, fov_vertical=180, batch_size=None):
        device = points.device
        batch_idx = points[:, 0].long()
        xyz = points[:, 1:4]
        r = torch.norm(xyz, dim=1)                                   # [N]
        phi = torch.atan2(xyz[:, 1], xyz[:, 0])                      # [-pi, pi]
        theta = torch.asin((xyz[:, 2] / r.clamp(min=1e-6)).clamp(-1, 1))

        # Continuous pixel coordinates.
        u_float = ((phi + math.pi) / (2 * math.pi)) * (W - 1)        # [0, W-1]
        half_fov = math.radians(fov_vertical) / 2
        v_float = ((theta + half_fov) / (2 * half_fov)) * (H - 1)    # [0, H-1]

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
