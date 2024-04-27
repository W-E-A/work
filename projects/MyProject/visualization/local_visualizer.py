from typing import Optional, Union, Tuple, List, Sequence
from mmdet3d.registry import VISUALIZERS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine.visualization.utils import tensor2ndarray, check_type
from mmengine.visualization.visualizer import Visualizer, VisBackendsType
from mmengine.dist import master_only
import numpy as np
import torch
from torch import Tensor
from .motion_visualization import plot_instance_map, visualise_output, plot_motion_prediction
import os
from PIL import Image

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
            np.round((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1]), # H
            np.round((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0]), # W
            np.round((self.pc_range[5] - self.pc_range[2]) / self.voxel_size[2]), # D
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
        bev_heads_voxel = np.round((bev_heads - self.offset_xy) / self.voxel_size[:2]).astype(np.int32) # N 2 2
        for c, h in zip(bev_corners_voxel, bev_heads_voxel):
            x = np.append(c[:, 0], c[0, 0])
            y = np.append(c[:, 1], c[0, 1])
            self.ax_save.plot(x, y, linewidth=2, **kwargs)
            self.ax_save.plot(h[:, 0], h[:, 1], linewidth=2, **kwargs)
    
    @master_only
    def draw_points(self,
                    positions: Union[np.ndarray, torch.Tensor],
                    colors: Union[str, tuple, List[str], List[tuple]] = 'g',
                    marker: Optional[str] = None,
                    sizes: Optional[Union[np.ndarray, torch.Tensor]] = None,
                    **kwargs):
        check_type('positions', positions, (np.ndarray, torch.Tensor))
        positions = tensor2ndarray(positions)

        if len(positions.shape) == 1:
            positions = positions[None]
        assert positions.shape[-1] == 2, (
            'The shape of `positions` should be (N, 2), '
            f'but got {positions.shape}')

        positions_voxel = np.round((positions - self.offset_xy) / self.voxel_size[:2]).astype(np.int32) # N 2

        self.ax_save.scatter(
            positions_voxel[:, 0], positions_voxel[:, 1], c=colors, s=sizes, marker=marker, **kwargs)
    
    @master_only
    def draw_arrows(self,
        positions: Union[np.ndarray, torch.Tensor],
        **kwargs
    ):
        check_type('positions', positions, (np.ndarray, torch.Tensor))
        positions = tensor2ndarray(positions)

        if len(positions.shape) == 1:
            positions = positions[None]
        assert positions.shape[-1] == 2, (
            'The shape of `positions` should be (N, 2), '
            f'but got {positions.shape}')

        positions_voxel = np.round((positions - self.offset_xy) / self.voxel_size[:2]).astype(np.int32) # N 2
        dx = positions_voxel[1][0] - positions_voxel[0][0]
        dy = positions_voxel[1][1] - positions_voxel[0][1]
        self.ax_save.arrow(positions_voxel[0][0], positions_voxel[0][1], dx, dy, **kwargs)

    @master_only
    def draw_texts(
        self,
        texts: Union[str, List[str]],
        positions: Union[np.ndarray, torch.Tensor],
        **kwargs
    ):
        check_type('positions', positions, (np.ndarray, torch.Tensor))
        positions = tensor2ndarray(positions)

        if len(positions.shape) == 1:
            positions = positions[None]
        assert positions.shape[-1] == 2, (
            'The shape of `positions` should be (N, 2), '
            f'but got {positions.shape}')

        positions_voxel = np.round((positions - self.offset_xy) / self.voxel_size[:2]).astype(np.int32) # N 2

        super().draw_texts(texts, positions_voxel, **kwargs)
    
    @master_only
    def draw_featmap(self, feat: Tensor, **kwargs):
        check_type('feat', feat, (np.ndarray, torch.Tensor))
        if isinstance(feat, np.ndarray):
            feat = torch.tensor(feat)
        self.set_image(super().draw_featmap(feat, **kwargs))
        
    @master_only
    def draw_motion_label(self, motion_label, save_dir: str, fps: int, display_order: str = 'vertical', gif: bool = True):
        video = visualise_output(labels=motion_label, output=None, display_order=display_order)[0]
        
        gifs = []
        for index in range(video.shape[0]):
            image = video[index].transpose((1, 2, 0))
            gifs.append(Image.fromarray(image))

        os.makedirs(save_dir, exist_ok=True)
        if gif:
            gifs[0].save(f"{save_dir}/motion_label.gif", save_all=True, append_images=gifs[1:], duration=1000 / fps, loop=0)
        else:
            for idx, img in enumerate(gifs):
                img.save(f"{save_dir}/motion_label_{idx}.png")
        
        # visualize BEV instance trajectory
        segmentation_binary = motion_label['segmentation']
        segmentation = segmentation_binary.new_zeros(
            segmentation_binary.shape).repeat(1, 1, 2, 1, 1)
        segmentation[:, :, 0] = (segmentation_binary[:, :, 0] == 0)
        segmentation[:, :, 1] = (segmentation_binary[:, :, 0] == 1)
        motion_label['segmentation'] = segmentation.float() * 10
        motion_label['instance_center'] = motion_label['centerness']
        motion_label['instance_offset'] = motion_label['offset']
        motion_label['instance_flow'] = motion_label['flow']
        figure_motion_label = plot_motion_prediction(motion_label)

        figure_motion_label = Image.fromarray(figure_motion_label)
        figure_motion_label.save(f"{save_dir}/motion_label_gt.png")

    @master_only
    def draw_motion_output(self, motion_output, save_dir: str, fps: int, display_order: str = 'vertical', gif: bool = True):
        video = visualise_output(labels=None, output=motion_output, display_order=display_order)[0]
        
        gifs = []
        for index in range(video.shape[0]):
            image = video[index].transpose((1, 2, 0))
            gifs.append(Image.fromarray(image))

        os.makedirs(save_dir, exist_ok=True)
        if gif:
            gifs[0].save(f"{save_dir}/motion_output.gif", save_all=True, append_images=gifs[1:], duration=1000 / fps, loop=0)
        else:
            for idx, img in enumerate(gifs):
                img.save(f"{save_dir}/motion_output_{idx}.png")
        
        figure_motion_pred = plot_motion_prediction(motion_output)

        figure_motion_pred = Image.fromarray(figure_motion_pred)
        figure_motion_pred.save(f"{save_dir}/motion_output_pred.png")

    # @master_only
    def just_save(self, save_path: str = "./bev_points.png"):
        self.fig_save.savefig(save_path, bbox_inches='tight', pad_inches=0) # type: ignore

    @master_only
    def clean(self):
        self.ax_save.cla()