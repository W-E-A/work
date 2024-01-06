from mmdet3d.registry import MODELS
from mmengine.model.base_module import BaseModule
from mmdet3d.models.middle_encoders.pillar_scatter import PointPillarsScatter
import math

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