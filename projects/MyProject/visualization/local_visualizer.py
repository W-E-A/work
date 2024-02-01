from typing import Optional, Union, Tuple, List, Sequence
from mmdet3d.registry import VISUALIZERS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine.visualization.utils import tensor2ndarray
from mmengine.visualization.visualizer import Visualizer, VisBackendsType
from mmengine.dist import master_only
import numpy as np
import torch
from torch import Tensor

@VISUALIZERS.register_module()
class SimpleLocalVisualizer(Visualizer):
    def __init__(
        self,
        pc_range,
        voxel_size,
        mask_range: Optional[List] = None,
        name='visualizer',
        vis_backends: VisBackendsType = None,
        save_dir: Optional[str] = None,
        points: Optional[Union[np.ndarray, str]] = None,
        point_color: Optional[Union[List[int], int]] = 255,
    ) -> None:
        super().__init__(
                name=name,
                image=None,
                vis_backends=vis_backends,
                save_dir=save_dir)
        
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.mask_range = mask_range
        self.grid_size = np.array([
            np.ceil((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1]), # H
            np.ceil((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0]), # W
            np.ceil((self.pc_range[5] - self.pc_range[2]) / self.voxel_size[2]), # D
        ]).astype(np.int32) # 1024 1024 1
        self.offset_xy = np.array([
            self.pc_range[0] + self.voxel_size[0] * 0.5,
            self.pc_range[1] + self.voxel_size[1] * 0.5
        ]).astype(np.float32)
        
        if isinstance(points, str):
            self.set_points_from_npz(points, point_color)
        elif isinstance(points, np.ndarray):
            self.set_points(points, point_color)

    @master_only
    def set_points(self, points, point_color: Optional[Union[List[int], int]] = 255):
        if isinstance(point_color, Sequence):
            point_colors = np.array(point_color, dtype=np.uint8)
        else:
            point_colors = np.array([point_color, point_color, point_color], dtype=np.uint8)
        points_xy = points[:, :2]
        if self.mask_range:
            keep_x = np.logical_or(
                    points_xy[:, 0] < self.mask_range[0],
                    points_xy[:, 0] > self.mask_range[3]
            )
            keep_y = np.logical_or(
                    points_xy[:, 1] < self.mask_range[1],
                    points_xy[:, 1] > self.mask_range[4]
            )
            keep_idx = np.logical_or(keep_x, keep_y)
            points_xy = points_xy[keep_idx]
        xy_voxel = np.clip(np.round((points_xy - self.offset_xy) / self.voxel_size[:2]).astype(np.int32), (0, 0), (self.grid_size[1]-1, self.grid_size[0]-1))
        map_vis = np.zeros((self.grid_size[0], self.grid_size[1], 3), dtype=np.uint8)
        map_vis[xy_voxel[:, 1], xy_voxel[:, 0]] = point_colors
        self.set_image(map_vis)
        self.ax_save.set_autoscale_on(False)

    @master_only
    def set_points_from_npz(self, file_path: str, **kwargs):
        assert file_path.endswith('npz')
        points = np.load(file_path)['data']
        self.set_points(points, **kwargs)

    @master_only
    def set_points_from_middle_feature(self, feat: Tensor):
        """
        CHW scatter feat
        """
        assert len(feat.shape) == 3
        map_vis = torch.sigmoid(torch.max(feat.permute(1, 2, 0), dim=-1, keepdim=True).values).detach().cpu().numpy()
        map_gray = np.clip(map_vis * 255, 0, 255).astype(np.uint8)
        map_gray = np.where(map_gray > 200, np.full_like(map_gray, 255), np.full_like(map_gray, 0))
        map_vis = np.tile(map_vis, (1, 1, 3))
        self.set_image(map_vis)

    @master_only
    def draw_bev_bboxes(self, boxes: LiDARInstance3DBoxes, **kwargs):
        assert isinstance(boxes, LiDARInstance3DBoxes)
        corners = tensor2ndarray(boxes.corners) # N 8 3
        bev_corners = corners[..., [0, 3, 7, 4], :2] # N 4 2 loop order
        bev_head_1 = corners[..., [5, 6], :2] # N 2 2
        bev_head_1 = (bev_head_1[:, 0] + bev_head_1[:, 1]) * 0.5 # N 2
        bev_head_2 = tensor2ndarray(boxes.gravity_center)[..., :2] # N 2
        bev_heads = np.stack([bev_head_1, bev_head_2], axis=1) # N 2 2
        bev_corners_voxel = np.round((bev_corners - self.offset_xy) / self.voxel_size[:2]).astype(np.int32) # N 4 2
        bev_heads_voxel = np.round((bev_heads - self.offset_xy) / self.voxel_size[:2]).astype(np.int32) # N 4 2
        for c, h in zip(bev_corners_voxel, bev_heads_voxel):
            x = np.append(c[:, 0], c[0, 0])
            y = np.append(c[:, 1], c[0, 1])
            self.ax_save.plot(x, y, linewidth=2, **kwargs)
            self.ax_save.plot(h[:, 0], h[:, 1], linewidth=2, **kwargs)

    @master_only
    def draw_bev_feat(self, feat: Tensor, **kwargs):
        self.set_image(self.draw_featmap(feat, **kwargs))

    @master_only
    def just_save(self, save_path: str = "./bev_points.png"):
        self.fig_save.savefig(save_path, bbox_inches='tight', pad_inches=0) # type: ignore

