"""
Pillar VFE, credits to OpenPCDet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        # [54390, 32, 10]
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(
                inputs[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False # type: ignore
        # [M, 64, 32]
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2,
                                                  1) if self.use_norm else x
        torch.backends.cudnn.enabled = True # type: ignore
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        # [M, 1, 64]
        # print(x_max.shape)
        

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(nn.Module):
    """
    pillar_vfe:
      use_norm: true
      with_distance: false
      use_absolute_xyz: true
      num_filters: [64]
    point_pillar_scatter:
      num_features: 64
    """
    def __init__(self, model_cfg, num_point_features, voxel_size,
                 point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg

        self.use_norm = self.model_cfg['use_norm']
        self.with_distance = self.model_cfg['with_distance']

        self.use_absolute_xyz = self.model_cfg['use_absolute_xyz']
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg['num_filters']
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters) # 4+6

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                         last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        # [M, ]
        # 32
        actual_num = torch.unsqueeze(actual_num, axis + 1) # [M, 1]
        # print(actual_num.shape)
        max_num_shape = [1] * len(actual_num.shape) # []
        max_num_shape[axis + 1] = -1
        # print(max_num_shape)
        # [32]
        max_num = torch.arange(max_num,
                               dtype=torch.int,
                               device=actual_num.device).view(max_num_shape)
        # print(max_num.shape)
        paddings_indicator = actual_num.int() > max_num
        # print(paddings_indicator.shape)
        return paddings_indicator

    def forward(self, batch_dict):
        """encoding voxel feature using point-pillar method
        Args:
            voxel_features: [M, 32, 4]
            voxel_num_points: [M,]
            voxel_coords: [M, 4]
        Returns:
            features: [M,64], after PFN
        """
        voxel_features, voxel_num_points, coords = \
            batch_dict['voxel_features'], batch_dict['voxel_num_points'], \
            batch_dict['voxel_coords']

        # [M, 32, 4]
        # [M, 4]
        # [M, ]
        # [M, 1, 3] / [M, 1, 1] 求出质心点
        points_mean = \
            voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
            voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        # [M, 32, 3] - [M, 1, 3] = [M, 32, 3] 每个点到质心点的距离
        f_cluster = voxel_features[:, :, :3] - points_mean

        # [M, 32, 3] 求每个点到体素中心点的距离
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        # 中心点X坐标 = 每个点减去（坐标*voxel大小还原到距离+最低位置偏移从而移动到中间）
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze( # X
                    1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze( # Y
                    1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze( # Z
                    1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, # type: ignore
                                     keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1) # [M, 32, 4+3+3]

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count,
                                           axis=0)

        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask # TODO ????
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features

        # import pdb
        # pdb.set_trace()

        return batch_dict
