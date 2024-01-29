from typing import Optional
from mmdet3d.registry import MODELS
from mmengine.model.base_module import BaseModule
from mmdet3d.models.middle_encoders.pillar_scatter import PointPillarsScatter
import math
import torch
import torch.nn as nn

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

        D = math.ceil((lidar_range[5] - lidar_range[2]) / voxel_size[2])
        H = math.ceil((lidar_range[4] - lidar_range[1]) / voxel_size[1])
        W = math.ceil((lidar_range[3] - lidar_range[0]) / voxel_size[0])

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
        gaussian_kernel = 1 / (2 * torch.pi * sigma) * torch.exp(-(torch.square(x) + torch.square(y)) / (2 * sigma**2))
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