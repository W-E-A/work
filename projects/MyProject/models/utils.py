from typing import Optional
from mmdet3d.registry import MODELS
from mmengine.model.base_module import BaseModule
from mmdet3d.models.middle_encoders.pillar_scatter import PointPillarsScatter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@MODELS.register_module()
class PointPillarsScatterWrapper(BaseModule):
    def __init__(self,
                 in_channels: int,
                 lidar_range: list,
                 voxel_size: list):
        super(PointPillarsScatterWrapper, self).__init__()
        self.in_channels = in_channels
        self.lidar_range = lidar_range
        self.voxel_size = voxel_size

        D = np.round((lidar_range[5] - lidar_range[2]) / voxel_size[2]).astype(np.int32)
        H = np.round((lidar_range[4] - lidar_range[1]) / voxel_size[1]).astype(np.int32)
        W = np.round((lidar_range[3] - lidar_range[0]) / voxel_size[0]).astype(np.int32)

        self.scatter = PointPillarsScatter(in_channels, [H, W])
    
    def forward(self, *args, **kwargs):
        return self.scatter( *args, **kwargs)


@MODELS.register_module()
class GaussianConv(BaseModule):
    def __init__(self,
                 kernel_size: int = 5,
                 sigma: float = 1.0,
                 thres: float = 1e-8,
                 impl: bool = True,
                 ):
        super(GaussianConv, self).__init__()
        self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
        self.thres = thres
        self.impl = impl
        device = self.gaussian_filter.weight.device
        dtype = self.gaussian_filter.weight.dtype
        center = kernel_size // 2
        x, y = torch.meshgrid(torch.linspace(0 - center, kernel_size // 2, kernel_size), torch.linspace(0 - center, kernel_size // 2, kernel_size))
        gaussian_kernel = 1 / (2 * np.pi * sigma) * torch.exp(-(torch.square(x) + torch.square(y)) / (2 * sigma**2))
        self.gaussian_filter.weight.data = gaussian_kernel.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.requires_grad = False # type: ignore

    def forward(self, x, thres: Optional[float] = None):
        if not thres:
            thres = self.thres
        if self.impl:
            smooth_x = self.gaussian_filter(x)
            smooth_x = torch.where(
                smooth_x > thres,
                torch.ones_like(smooth_x, device=x.device, dtype=x.dtype),
                torch.zeros_like(smooth_x, device=x.device, dtype=x.dtype)
            )
        else:
            smooth_x = x
        return smooth_x


@MODELS.register_module()
class BevFeatureSlicer(BaseModule):
    # crop the interested area in BEV feature for semantic map segmentation
    def __init__(self, grid_conf, new_grid_conf):
        super(BevFeatureSlicer, self).__init__()

        if grid_conf == new_grid_conf:
            self.identity_mapping = True
        else:
            self.identity_mapping = False
            pc_range = grid_conf[0]
            voxel_size = grid_conf[1]
            new_pc_range = new_grid_conf[0]
            new_voxel_size = new_grid_conf[1]

            bev_start_position = torch.tensor([
                pc_range[0] + voxel_size[0]/2, # W X
                pc_range[1] + voxel_size[1]/2, # H Y
                pc_range[2] + voxel_size[2]/2, # D Z
            ])
            bev_dimension = torch.tensor([
                (pc_range[3] - pc_range[0]) / voxel_size[0], # W X
                (pc_range[4] - pc_range[1]) / voxel_size[1], # H Y
                (pc_range[5] - pc_range[2]) / voxel_size[2], # D Z
            ], dtype=torch.long)

            new_resolution = torch.tensor(
                [new_voxel_size[0],new_voxel_size[1],new_voxel_size[2]])
            new_start_position = torch.tensor([
                new_pc_range[0] + new_voxel_size[0]/2, # W X
                new_pc_range[1] + new_voxel_size[1]/2, # H Y
                new_pc_range[2] + new_voxel_size[2]/2, # D Z
            ])

            self.map_x = torch.arange(
                new_start_position[0], new_pc_range[3], new_resolution[0])

            self.map_y = torch.arange(
                new_start_position[1], new_pc_range[4], new_resolution[1])

            # convert to normalized coords
            self.norm_map_x = self.map_x / (- bev_start_position[0])
            self.norm_map_y = self.map_y / (- bev_start_position[1])

            norm_map_xx,norm_map_yy = torch.meshgrid(self.norm_map_x, self.norm_map_y)
            self.map_grid = torch.stack((norm_map_xx.transpose(1,0), norm_map_yy.transpose(1,0)), dim=2)

    def forward(self, x):
        # x: bev feature map tensor of shape (b, c, h, w)
        if self.identity_mapping:
            return x
        else:
            grid = self.map_grid.unsqueeze(0).type_as(
                x).repeat(x.shape[0], 1, 1, 1)

            return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)