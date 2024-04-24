from typing import List, Optional, Union,  Dict, Tuple
from numpy import ndarray
from mmdet3d.registry import TRANSFORMS, VISUALIZERS
from mmdet3d.datasets.transforms.loading import LoadPointsFromFile, LoadAnnotations3D
from mmcv.transforms.base import BaseTransform
import numpy as np
from mmengine.structures import BaseDataElement
from mmdet3d.structures import LiDARInstance3DBoxes, CameraInstance3DBoxes, DepthInstance3DBoxes, Det3DDataSample, LiDARPoints, Coord3DMode
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from os import path as osp
import copy
import cv2
from ..utils import calc_relative_pose, convert_instance_mask_to_center_and_offset_label
import matplotlib.pyplot as plt
import torch
import os
from mmengine.logging import print_log
import logging
import time

# NOTE add all ego motion 6 DOF
# NOTE 把现在和未来的筛选放这里，intput 提供历史+当前输入，而data_samples提供未来+当前标签
# NOTE 原文将box提前转换到EGO坐标系，数据处理方面，速度矢量在制作pkl的时候反转Y轴，然后统一随着box转到EGO坐标系下
# NOTE 这里保持主干的检测在lidar坐标系下，而前后处理需要转换坐标自行从lidar转换出来
# FIXME test的时候加上场景的详细信息
# NOTE 添加其他代理到ego的变换矩阵
# NOTE 原文V2X产生的数据是LIST，每个元素代表一个Agent，第一个是EGO，只有EGO包含了场景的所有信息，EGO单独经过
#       pipline处理，最后将所有的结果按照每一个Agent的KEY整理为KEY:LIST的形式也就是提前collect

# @TRANSFORMS.register_module()
# class AddPoseNoise(BaseTransform):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#     def transform(self, results: Dict) -> Union[Dict,Tuple[List, List],None]:
#         return None

### Single Agent Pipline ###

@TRANSFORMS.register_module()
class LoadPointsNPZ(LoadPointsFromFile):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _load_points(self, pts_filename: str) -> ndarray:
        assert osp.exists(pts_filename) and pts_filename.endswith('.npz')
        pc = np.load(pts_filename)['data']
        return pc


@TRANSFORMS.register_module()
class InnerPointsRangeFilter(BaseTransform):
    def __init__(self, point_cloud_range: List[float]) -> None:
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def transform(self, input_dict: dict) -> dict:
        points = input_dict['points']
        points_mask = points.in_range_3d(self.pcd_range)
        points_mask = torch.logical_not(points_mask)
        clean_points = points[points_mask]
        input_dict['points'] = clean_points
        points_mask = points_mask.numpy()

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[points_mask]

        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[points_mask]

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@TRANSFORMS.register_module()
class LoadAnnotations3DV2X(LoadAnnotations3D):
    def __init__(self,
                 *args,
                 with_bbox_3d_isvalid: bool = True,
                 with_track_id: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.with_bbox_3d_isvalid = with_bbox_3d_isvalid
        self.with_track_id = with_track_id

    def _load_bbox_3d_isvalid(self, results: dict) -> dict:
        results['bbox_3d_isvalid'] = results['ann_info']['bbox_3d_isvalid']
        return results
    
    def _load_track_id(self, results: dict) -> dict:
        results['track_id'] = results['ann_info']['track_id']
        return results

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        if self.with_bbox_3d_isvalid:
            results = self._load_bbox_3d_isvalid(results)
        if self.with_track_id:
            results = self._load_track_id(results)
        return results

    def __repr__(self) -> str:
        repr_str = super().__repr__()
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d_isvalid={self.with_bbox_3d_isvalid}, '
        repr_str += f'{indent_str}with_track_id={self.with_track_id}, '

        return repr_str


@TRANSFORMS.register_module()
class ConstructEGOBox(BaseTransform):
    def __init__(self, ego_id: int = -100) -> None:
        self.ego_id = ego_id

    def transform(self, input_dict: Dict) -> Union[Dict,Tuple[List, List],None]:
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        track_id = input_dict['track_id']
        ego_box = gt_bboxes_3d.tensor[track_id == self.ego_id]
        if ego_box.shape[0] != 1 and input_dict['agent'] != 'infrastructure':
            box_type_3d = input_dict['box_type_3d']
            box_mode_3d = input_dict['box_mode_3d']
            ego_height = gt_bboxes_3d.tensor[0, 2]
            temp_box = torch.zeros_like(gt_bboxes_3d.tensor[0])
            temp_box[2] = ego_height
            temp_box[3] = 3.99
            temp_box[4] = 1.85
            temp_box[5] = 1.62
            ego_bboxes_3d = box_type_3d(
                temp_box.unsqueeze(0),
                box_dim=temp_box.shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d)
            gt_bboxes_3d = LiDARInstance3DBoxes.cat([gt_bboxes_3d, ego_bboxes_3d])
            input_dict['gt_bboxes_3d'] = gt_bboxes_3d
            input_dict['track_id'] = np.append(input_dict['track_id'], self.ego_id)
            input_dict['gt_labels_3d'] = np.append(input_dict['gt_labels_3d'], 0)
            input_dict['bbox_3d_isvalid'] = np.append(input_dict['bbox_3d_isvalid'], False)
        return input_dict
    

@TRANSFORMS.register_module()
class ObjectRangeFilterV2X(BaseTransform):

    def __init__(self, point_cloud_range: List[float]) -> None:
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def transform(self, input_dict: dict) -> dict:
        """Transform function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
            keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]] # type: ignore
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]] # type: ignore

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        bbox_3d_isvalid = input_dict['bbox_3d_isvalid']
        track_id = input_dict['track_id']
        mask = gt_bboxes_3d.in_range_bev(bev_range) # type: ignore
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(bool)]
        bbox_3d_isvalid = bbox_3d_isvalid[mask.numpy().astype(bool)]
        track_id = track_id[mask.numpy().astype(bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d
        input_dict['bbox_3d_isvalid'] = bbox_3d_isvalid
        input_dict['track_id'] = track_id

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@TRANSFORMS.register_module()
class ObjectNameFilterV2X(BaseTransform):
    """Filter GT objects by their names.

    Required Keys:

    - gt_labels_3d

    Modified Keys:

    - gt_labels_3d

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes: List[str]) -> None:
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def transform(self, input_dict: dict) -> dict:
        """Transform function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
            keys are updated in the result dict.
        """
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=bool)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]
        input_dict['bbox_3d_isvalid'] = input_dict['bbox_3d_isvalid'][gt_bboxes_mask]
        input_dict['track_id'] = input_dict['track_id'][gt_bboxes_mask]

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str


@TRANSFORMS.register_module()
class ObjectTrackIDFilter(BaseTransform):
    def __init__(self, ids: List[str], impl: bool = False) -> None:
        self.ids = ids
        self.impl = impl

    def transform(self, input_dict: dict) -> dict:
        track_id = input_dict['track_id']
        track_id_isvalid = np.array([id not in self.ids for id in track_id], dtype=bool)        

        if self.impl:
            input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][track_id_isvalid]
            input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][track_id_isvalid]
            input_dict['bbox_3d_isvalid'] = input_dict['bbox_3d_isvalid'][track_id_isvalid]
            input_dict['track_id'] = input_dict['track_id'][track_id_isvalid]
        else:
            input_dict['bbox_3d_isvalid'] = track_id_isvalid

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(ids={self.ids})'
        repr_str += f'(impl={self.impl})'
        return repr_str


@TRANSFORMS.register_module()
class ObjectValidFilter(BaseTransform):
    def __init__(self, impl: bool = False) -> None:
        self.impl = impl

    def transform(self, input_dict: dict) -> dict:
        bbox_3d_isvalid = input_dict['bbox_3d_isvalid']

        if self.impl:
            input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][bbox_3d_isvalid]
            input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][bbox_3d_isvalid]
            input_dict['bbox_3d_isvalid'] = input_dict['bbox_3d_isvalid'][bbox_3d_isvalid]
            input_dict['track_id'] = input_dict['track_id'][bbox_3d_isvalid]

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(impl={self.impl})'
        return repr_str


@TRANSFORMS.register_module()
class Pack3DDetInputsV2X(Pack3DDetInputs):
    INPUTS_KEYS = ['points', 'img']
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels', 'depths', 'centers_2d', 'bbox_3d_isvalid', 'track_id'
    ]
    INSTANCEDATA_2D_KEYS = [
        'gt_bboxes',
        'gt_bboxes_labels',
    ]
    SEG_KEYS = [
        'gt_seg_map', 'pts_instance_mask', 'pts_semantic_mask',
        'gt_semantic_seg'
    ]
    def __init__(self,
                 *args,
                 meta_keys: tuple = (
                    'img_path', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img', 'pad_shape',
                    'img_norm_cfg', 'trans_mat', 'affine_aug', 'sweep_img_metas', 'ori_cam2img', 'cam2global',
                    'crop_offset', 'img_crop_offset', 'resize_img_shape', 'lidar2cam', 'ori_lidar2img',
                    'num_ref_frames', 'num_views',
                    'scale_factor', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 'pcd_trans',
                    'pcd_scale_factor', 'pcd_rotation', 'pcd_rotation_angle', 'transformation_3d_flow',
                    'box_mode_3d', 'box_type_3d', 'num_pts_feats', 'lidar_path',
                    'sample_idx', 'ego2global', 'axis_align_matrix',
                    'lidar2ego', 'bev_path', 'vehicle_speed_x', 'vehicle_speed_y'
                ),
                **kwargs) -> None:
        super().__init__(*args, meta_keys = meta_keys, **kwargs)
        

    def transform(self, results: Dict) -> Union[Dict,Tuple[List, List],None]:
        return super().transform(results) # type: ignore

### Scene Pipline ###

@TRANSFORMS.register_module()
class GatherV2XPoseInfo(BaseTransform):
    """
    row ij j to i
    col ji i to j

    motion rela to seq start

    more: motion rela to last frame FIXME
    """
    def transform(self, input_dict: Dict) -> Union[Dict,Tuple[List, List],None]:
        co_length = input_dict['co_length']
        seq_length = input_dict['seq_length']
        example_seq = input_dict['example_seq']
        present_idx = input_dict['present_idx']
        future_seq = range(seq_length)[present_idx:] # only future motion

        seq_pose_matrix = []
        seq_loc_matrix = []
        for i, data_info_list in enumerate(example_seq):
            co_pose_matrix = []
            ego_loc_matrix = []
            for j in range(co_length):
                ego2global = np.array(data_info_list[j]['data_samples'].metainfo['lidar2ego'], dtype=np.float32) @ np.array(data_info_list[j]['data_samples'].metainfo['ego2global'], dtype=np.float32)
                other2global = [np.array(data_info_list[k]['data_samples'].metainfo['lidar2ego'], dtype=np.float32) @ np.array(data_info_list[k]['data_samples'].metainfo['ego2global'], dtype=np.float32) for k in range(co_length)]
                co_pose_matrix.append(calc_relative_pose(ego2global, other2global)) # type: ignore
                ego_loc_matrix.append(ego2global)
            co_pose_matrix = np.stack(co_pose_matrix, axis=0)
            ego_loc_matrix = np.stack(ego_loc_matrix, axis=0)
            seq_pose_matrix.append(co_pose_matrix)
            seq_loc_matrix.append(ego_loc_matrix)
        seq_pose_matrix = np.stack(seq_pose_matrix, axis=0)
        seq_loc_matrix = np.stack(seq_loc_matrix, axis=0)
        input_dict['pose_matrix'] = seq_pose_matrix # seq, co, co, 4, 4
        input_dict['loc_matrix'] = seq_loc_matrix # seq, co, 4, 4

        future_motion_matrix = []
        future_motion_rela_matrix = []
        for j in range(co_length):
            future_motion_list = [np.array(example_seq[i][j]['data_samples'].metainfo['ego2global'], dtype=np.float32) for i in future_seq]
            temp = [np.eye(4, dtype=np.float32)]
            temp.extend(list(map(calc_relative_pose, future_motion_list[1:], future_motion_list[:-1])))
            future_motion_rela_matrix.append(np.stack(temp, axis=0))
            future_motion_matrix.append(np.stack(calc_relative_pose(future_motion_list[0], future_motion_list), axis=0)) # type: ignore
        future_motion_matrix = np.stack(future_motion_matrix, axis=0) # NOTE starts with identity
        future_motion_rela_matrix = np.stack(future_motion_rela_matrix, axis=0) # NOTE starts with identity
        input_dict['future_motion_matrix'] = future_motion_matrix # co seq 4 4 
        input_dict['future_motion_rela_matrix'] = future_motion_rela_matrix # co seq 4 4 
        return input_dict


@TRANSFORMS.register_module()
class MakeMotionLabels(BaseTransform):


    def __init__(self,
        pc_range,
        voxel_size,
        ego_id: int = -100,
        only_vehicle: bool = False,
        vehicle_id_list: list = [0, 1, 2],
        filter_invalid: bool = True,
        ignore_index:int = 255,
        visualizer_cfg: dict = None,
        just_save_root: str = None,
        ) -> None:
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.voxel_size = np.array(self.voxel_size).astype(np.float32)
        self.grid_size = np.array([
            np.round((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1]), # H
            np.round((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0]), # W
            np.round((self.pc_range[5] - self.pc_range[2]) / self.voxel_size[2]), # D
        ]).astype(np.int32)
        self.offset_xy = np.array([
            self.pc_range[0] + self.voxel_size[0] * 0.5,
            self.pc_range[1] + self.voxel_size[1] * 0.5
        ]).astype(np.float32)
        self.warp_size = (0.5 * (self.pc_range[3] - self.pc_range[0]), 0.5 * (self.pc_range[4] - self.pc_range[1]))
        self.ego_id = ego_id
        self.only_vehicle = only_vehicle
        self.vehicle_id_list = vehicle_id_list
        self.filter_invalid = filter_invalid
        self.ignore_index = ignore_index
        if visualizer_cfg is not None:
            self.visualizer: SimpleLocalVisualizer = VISUALIZERS.build(visualizer_cfg)
            assert just_save_root != None
            os.makedirs(just_save_root, exist_ok=True)
        self.just_save_root = just_save_root

    def transform(self, input_dict) -> Union[Dict,Tuple[List, List],None]:
        sample_idx = input_dict['sample_idx'] # 采样序列的唯一ID
        sample_interval = input_dict['sample_interval'] # 序列采集间隔
        present_idx = input_dict['present_idx'] # 当前序列第几位开始是关键帧, 比如0
        seq_length = input_dict['seq_length'] # 当前序列长度，比如6
        scene_name = input_dict['scene_name'] # 序列所属场景
        seq_timestamps = input_dict['seq_timestamps'] # 序列时间戳
        co_agents = copy.deepcopy(input_dict['co_agents']) # 参与协同的代理名称
        example_seq = input_dict['example_seq'] # 经过单车pipline得到的输入和标签数据
        future_seq_position = range(seq_length)[present_idx:] # 取出未来帧在序列中的位置， 比如3, 4, 5
        future_length = len(future_seq_position) # 表明未来帧的长度，比如3
        # future_pose_matrix = input_dict['pose_matrix'][present_idx:, ...] # x c c 4 4 # 未来帧的每帧变换关系
        # future_motion_matrix = input_dict['future_motion_matrix'] # c x 4 4 未来帧相对于起始帧的位姿关系
        # future_loc_matrix = input_dict['loc_matrix'][present_idx:, :, ...] # c x 4 4 # 未来帧ego的实际位置
        future_motion_rela_matrix = input_dict['future_motion_rela_matrix'] # c x 4 4 未来帧对于前一帧的位姿关系

        for j, agent in enumerate(co_agents):
            
            track_map = {}
            segmentations = []
            instances = []
            for i, timestamp in enumerate(future_seq_position):
                gt_instances_3d = example_seq[timestamp][j]['data_samples'].gt_instances_3d.clone()

                if gt_instances_3d.bboxes_3d == None:
                    segmentation = np.ones(
                        (self.grid_size[0], self.grid_size[1])) * self.ignore_index # H, W
                    instance = np.ones_like(segmentation) * self.ignore_index
                else:
                    segmentation = np.zeros((self.grid_size[0], self.grid_size[1])) # H, W
                    instance = np.zeros_like(segmentation)    

                if self.only_vehicle:
                    vehicle_mask = np.isin(gt_instances_3d.labels_3d, self.vehicle_id_list)
                    gt_instances_3d = gt_instances_3d[vehicle_mask]
                
                if agent != 'infrastructure':
                    ego_mask = gt_instances_3d.track_id != self.ego_id
                    gt_instances_3d = gt_instances_3d[ego_mask]
                
                if self.filter_invalid:
                    gt_instances_3d = gt_instances_3d[gt_instances_3d.bbox_3d_isvalid]

                bbox_corners = gt_instances_3d.bboxes_3d.corners[:, [0, 3, 7, 4], :2].numpy() # four corner B, 4, 2
                bbox_corners_voxel = np.round((bbox_corners - self.offset_xy) / self.voxel_size[:2]).astype(np.int32) # N 4 2 to voxel coor

                for index, id in enumerate(gt_instances_3d.track_id):
                    if id not in track_map:
                        track_map[id] = len(track_map) + 1
                    instance_id = track_map[id]
                    poly_region = bbox_corners_voxel[index]
                    cv2.fillPoly(segmentation, [poly_region], 1.0) # 语义分割为01
                    cv2.fillPoly(instance, [poly_region], instance_id) # 实例分割为0~254

                segmentations.append(segmentation)
                instances.append(instance)

            segmentations = np.stack(segmentations, axis=0).astype(np.int32) # [fu_len, H, W]
            instances = np.stack(instances, axis=0).astype(np.int32) # [fu_len, H, W]

            instance_centerness, instance_offset, instance_flow = convert_instance_mask_to_center_and_offset_label(
                instances,
                future_motion_rela_matrix[j],
                len(track_map),
                ignore_index = self.ignore_index,
                spatial_extent = self.warp_size,
            ) # len, 1, h, w  len, 2, h, w  len, 2, h, w

            # 如果分割有任何是没有box的则清除中心度为ignore
            invalid_mask = (segmentations[:, 0, 0] == self.ignore_index)
            instance_centerness[invalid_mask] = self.ignore_index
            motion_label = {
                'motion_segmentation': torch.from_numpy(segmentations), # [fu_len, H, W]
                'motion_instance': torch.from_numpy(instances), # [fu_len, H, W]
                'instance_centerness': torch.from_numpy(instance_centerness), # len, 1, h, w
                'instance_offset': torch.from_numpy(instance_offset), # len, 2, h, w
                'instance_flow': torch.from_numpy(instance_flow), # len, 2, h, w
            }
            example_seq[present_idx][j]['motion_label'] = motion_label
            if self.visualizer:
                save_path = os.path.join(self.just_save_root, f'{sample_idx}', f'{agent}')
                os.makedirs(save_path, exist_ok=True)

                for ts, instance in enumerate(instances):
                    self.visualizer.draw_instance_label(instance, ignore_index = self.ignore_index)
                    self.visualizer.just_save(os.path.join(save_path, f"instance_{ts}.png"))
                    self.visualizer.clean()
        return input_dict


def log(msg = "" ,level: int = logging.INFO):
    print_log(msg, "current", level)
@TRANSFORMS.register_module()
class CorrelationFilter(BaseTransform):

    def __init__(self,
        ego_name: str = 'ego_vehicle',
        with_velocity: bool = True,
        only_vehicle: bool = False,
        vehicle_id_list: list = [0, 1, 2],
        ego_id: int = -100,
        min_distance_thres: float = 5,
        max_distance_thres: float = 20,
        alpha_coeff: float = 1,
        beta_coeff: float = 1,
        gamma_coeff: float = 2,
        visualizer_cfg: dict = None,
        just_save_root: str = None,
        increment_save: bool = True,
        verbose: bool = False
    ) -> None:
        self.ego_name = ego_name
        self.with_velocity = with_velocity
        self.only_vehicle = only_vehicle
        self.vehicle_id_list = vehicle_id_list
        self.ego_id = ego_id
        self.min_distance_thres = min_distance_thres
        self.max_distance_thres = max_distance_thres
        self.alpha_coeff = alpha_coeff
        self.beta_coeff = beta_coeff
        self.gamma_coeff = gamma_coeff
        self.visualizer = None
        if visualizer_cfg is not None:
            self.visualizer: SimpleLocalVisualizer = VISUALIZERS.build(visualizer_cfg)
            assert just_save_root != None
            os.makedirs(just_save_root, exist_ok=True)
        self.just_save_root = just_save_root
        self.increment_save = increment_save
        self.verbose = verbose
        
        self.cmaps = [
            'Oranges', 'Greens', 'Purples', 'Oranges', 'PuRd', 'BuPu',
        ]
        # self.cmaps = [
        #     'Greys', 'Purples', 'Greens', 'Oranges', 'Reds',
        #     'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        #     'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
        # ]

    def linear_motion_score(self,
            ego_center: np.ndarray,
            target_center: np.ndarray,
            min_distance_thres: float,
            max_distance_thres: float,
            alpha_coeff: float # 这个由网络轨迹预测给出轨迹的概率p，然后让(1 - p)表示如果轨迹概率为1，则少衰减比如0.5，概率接近0则多衰减比如2 ？？？？？
        ):
        assert ego_center.shape == target_center.shape
        assert len(ego_center.shape) == 2
        if len(ego_center) <= 0:
            return 0
        
        encounter_distance = np.linalg.norm(ego_center - target_center, axis=1)
        encounter_distance = np.clip(encounter_distance, min_distance_thres, max_distance_thres)
        base_score = np.interp(encounter_distance, [min_distance_thres, max_distance_thres], [1, 0])
        time_decay_coeff = np.exp(-alpha_coeff * np.linspace(0, 1, len(base_score)))
        time_decay_coeff = time_decay_coeff / np.sum(time_decay_coeff)
        decay_score = time_decay_coeff * base_score
        final_score = np.sum(decay_score)

        return final_score

    def potential_score(self,
        ego_center: np.ndarray, # 2,
        ego_vel: np.ndarray, # 2,
        target_center: np.ndarray, # 2,
        target_vel: np.ndarray, # 2,
        target_label,
        min_distance_thres: float,
        beta_coeff: float, 
        gamma_coeff: float,
        weight = [1, 1, 1, 1, 1, 1, ]
    ):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        assert ego_center.shape == target_center.shape == ego_vel.shape == target_vel.shape
        assert len(ego_center) == 2 and len(ego_center.shape) == 1
            
        rela_vector = ego_center - target_center
        rela_vector_norm = np.linalg.norm(rela_vector, ord=2)
        rela_vector_norm = np.clip(rela_vector_norm, min_distance_thres, np.inf)
        rela_vel = target_vel - ego_vel
        rela_vel_norm = np.linalg.norm(rela_vel, ord=2)
        rela_vel_norm = np.clip(rela_vel_norm, 1e-6, np.inf)
        ctheta = np.dot(rela_vector, rela_vel) / rela_vector_norm / rela_vel_norm
        
        potential = weight[target_label] * np.exp(beta_coeff * rela_vel_norm * ctheta) / np.power(rela_vector_norm, gamma_coeff)
        p_score = sigmoid(potential) - 0.5
        return potential, p_score


    def transform(self, input_dict) -> Union[Dict,Tuple[List, List],None]:
        sample_idx = input_dict['sample_idx'] # 采样序列的唯一ID
        sample_interval = input_dict['sample_interval'] # 序列采集间隔
        present_idx = input_dict['present_idx'] # 当前序列第几位开始是关键帧, 比如0
        seq_length = input_dict['seq_length'] # 当前序列长度，比如6
        scene_name = input_dict['scene_name'] # 序列所属场景
        seq_timestamps = input_dict['seq_timestamps'] # 序列时间戳
        co_agents = copy.deepcopy(input_dict['co_agents']) # 参与协同的代理名称
        example_seq = input_dict['example_seq'] # 经过单车pipline得到的输入和标签数据
        future_seq_position = range(seq_length)[present_idx:] # 取出未来帧在序列中的位置， 比如3, 4, 5
        future_length = len(future_seq_position) # 表明未来帧的长度，比如3
        future_pose_matrix = input_dict['pose_matrix'][present_idx:, ...] # x c c 4 4 # 未来帧的每帧变换关系
        future_motion_matrix = input_dict['future_motion_matrix'] # c x 4 4 未来帧相对于起始帧的位姿关系
        future_loc_matrix = input_dict['loc_matrix'][present_idx:, :, ...] # c x 4 4 # 未来帧ego的实际位置

        if self.verbose:
            log(scene_name)
            log(seq_timestamps)

        co_agents = {v : k for k, v in enumerate(co_agents)} # str : int
        assert self.ego_name in co_agents and self.ego_name != 'infrastructure'
        track_map_list = []
        for agent, j in co_agents.items():
            if agent == self.ego_name:
                # 如果处理的是ego则先得到ego的规划轨迹和关键帧相对于其他协同对象的变换矩阵
                ego_motion_matrix = future_motion_matrix[j] # x 4 4 rela to present
                present_trans_matrix = future_pose_matrix[0, :, j] # c 4 4
                present_trans_matrix = np.delete(present_trans_matrix, j, axis=0) # ego to [ego, other, ...] at present c-1 4 4
                if self.with_velocity:
                    ego_present_vel = np.array(
                        [example_seq[present_idx][j]['data_samples'].metainfo['vehicle_speed_x'],
                         example_seq[present_idx][j]['data_samples'].metainfo['vehicle_speed_y']],
                         dtype=np.float32
                    )
                    if self.verbose:
                        log(f"ego : {ego_present_vel}")
            else:
                # 如果处理的是协同代理，则对于每个代理在时序上跟踪所有目标，相对于关键帧的轨迹，每个代理得到一个track_map
                track_map = {}
                for i, timestamp in enumerate(future_seq_position):
                    gt_instances_3d = example_seq[timestamp][j]['data_samples'].gt_instances_3d.clone()

                    if self.only_vehicle:
                        vehicle_mask = np.isin(gt_instances_3d.labels_3d, self.vehicle_id_list)
                        gt_instances_3d = gt_instances_3d[vehicle_mask]
                    
                    if agent != 'infrastructure':
                        ego_mask = gt_instances_3d.track_id != self.ego_id
                        gt_instances_3d = gt_instances_3d[ego_mask]

                    # gt_instances_3d = gt_instances_3d[gt_instances_3d.bbox_3d_isvalid] # only visible target bug do not modify

                    lidar2agent = np.array(example_seq[timestamp][j]['data_samples'].metainfo['lidar2ego'], dtype=np.float32) # type: ignore
                    agent2present = future_motion_matrix[j][i]
                    lidar2present = lidar2agent @ agent2present
                    gt_instances_3d.bboxes_3d.rotate(lidar2present[:3, :3].T, None)
                    gt_instances_3d.bboxes_3d.translate(lidar2present[:3, 3]) # valid boxes to present
                    
                    for idx, id in enumerate(gt_instances_3d.track_id):
                        bboxes_xy = gt_instances_3d.bboxes_3d.tensor[idx].numpy()[:2] # present坐标系下其他目标的present的中心
                        bboxes_vel = gt_instances_3d.bboxes_3d.tensor[idx].numpy()[-2:] # present velocity
                        label = gt_instances_3d.labels_3d[idx]
                        if id not in track_map:
                            track_map[id] = {
                                # 'color': self.cmaps[np.random.randint(0, len(self.cmaps))],
                                # 'color': self.cmaps[label],
                                'label': label,
                                'center': [],
                                'vel': [],
                                'start': i
                            }
                        track_map[id]['center'].append(bboxes_xy)
                        if self.with_velocity:
                            track_map[id]['vel'].append(bboxes_vel)
                track_map_list.append(track_map)

        ego_dix = co_agents.pop(self.ego_name) # ignore the ego

        for i, (agent, j) in enumerate(co_agents.items()): # agent
            # 对于每个协同代理的视角下，将ego的规划轨迹转到协同代理的当前坐标系下
            track_map = track_map_list[i]
            present_instances_3d = example_seq[present_idx][j]['data_samples'].gt_instances_3d
            trans = present_trans_matrix[i] # type: ignore
            rela_matrix = np.stack([trans @ ego_motion_matrix[k] for k in range(future_length)], axis=0) # type: ignore
            rela_centers = rela_matrix[:, :2, 3] # x 2
            if self.with_velocity:
                rots = np.linalg.inv(future_loc_matrix[j, 0, ...])[:2, :2]# 每个ego速度对于世界，先转到对于present的代理描述下
                trans_rot = trans[:2, :2] # ego和代理之间具有相对运动，参考上述的motion通过当前的trans转到代理下
                rela_vel = ego_present_vel @ rots.T @ trans_rot.T

            if self.visualizer:
                self.visualizer.set_points_from_npz(example_seq[present_idx][j]['data_samples'].metainfo["lidar_path"])
            
            # 对于每个协同代理的检测结果（这里是真值），计算其和ego轨迹的关系
            result_dict = {}
            if self.with_velocity:
                potential_dict = {}
            for id, v in track_map.items():
                # 对于每一条轨迹
                # color = v['color'] # 类型颜色
                start = v['start'] # 开始跟踪的时间戳
                centers = np.stack(v['center'], axis=0) # N, 2 # 轨迹中心点
                cmp_centers = rela_centers[start:start + len(centers)] # N, 2 这里保证ego是从头开始的，因此在有效预测目标轨迹的时间段对比
                correlation_score = 0
                motion_score = self.linear_motion_score(
                    cmp_centers, centers, self.min_distance_thres, self.max_distance_thres, self.alpha_coeff
                )
                if self.with_velocity:
                    p_score = 0
                    if start == 0: # 忽略当前之后再出现的目标的速度
                        vel = v['vel'][start]
                        label = v['label']
                        if self.verbose:
                            log(f"{id} : {vel}")
                        potential, p_score = self.potential_score(
                            cmp_centers[0],
                            rela_vel,
                            centers[0],
                            vel,
                            label,
                            self.min_distance_thres,
                            self.beta_coeff,
                            self.gamma_coeff
                        )
                        correlation_score = motion_score * 0.5 + p_score
                else:
                    correlation_score = motion_score
                result_dict[id] = correlation_score
                if self.with_velocity:
                    potential_dict[id] = p_score # potential

                if self.visualizer:
                    if self.with_velocity and start == 0:
                        vel_vector = np.stack([centers[0], centers[0] + sample_interval * 0.1 * vel]) # 2, 2
                        self.visualizer.draw_arrows(vel_vector, fc = 'r', ec = 'r', width = 6, head_width = 12, head_length = 10)
                    self.visualizer.draw_points(centers, colors=np.linspace(0.0, 1.0, len(centers))[::-1], sizes = 40, cmap='Greens')
            
            # import pdb
            # pdb.set_trace()
            correlation_scores = np.array([result_dict[id] if id in result_dict else 0 for id in present_instances_3d.track_id], dtype=np.float32)
            potentials = np.array([potential_dict[id] if id in potential_dict else 0 for id in present_instances_3d.track_id], dtype=np.float32)
            
            present_instances_3d['correlation'] = torch.tensor(correlation_scores, dtype=torch.float32)
            example_seq[present_idx][j]['data_samples'].gt_instances_3d = present_instances_3d
            if self.visualizer:
                self.visualizer.draw_points(rela_centers, colors=np.linspace(0.0, 1.0, future_length)[::-1], sizes = 80, cmap='Blues')
                if self.with_velocity:
                    rela_vel_vector = np.stack([rela_centers[0], rela_centers[0] + sample_interval * 0.1 * rela_vel])
                    self.visualizer.draw_arrows(rela_vel_vector, fc = 'm', ec = 'm', width = 6, head_width = 12, head_length = 10)
                self.visualizer.draw_bev_bboxes(present_instances_3d.bboxes_3d, c='#FF8000')
                correlation_scores_str = [f"{s:.2f}" for s in list(correlation_scores)]
                self.visualizer.draw_texts(
                    correlation_scores_str,
                    present_instances_3d.bboxes_3d.gravity_center[..., :2],
                    font_sizes = 12,
                    colors = '#FFFF00',
                    vertical_alignments = 'top',
                    horizontal_alignments = 'left')
                potentials_str = [f"{s:.2f}" for s in list(potentials)]
                self.visualizer.draw_texts(
                    potentials_str,
                    present_instances_3d.bboxes_3d.gravity_center[..., :2],
                    font_sizes = 12,
                    colors = '#00FFFF',
                    vertical_alignments = 'top',
                    horizontal_alignments = 'right')
                self.visualizer.draw_texts(
                    present_instances_3d.track_id.tolist(),
                    present_instances_3d.bboxes_3d.gravity_center[..., :2],
                    font_sizes = 12,
                    colors = '#FF00FF',
                    vertical_alignments = 'bottom',
                    horizontal_alignments = 'right')

                if self.verbose:
                    log(result_dict)
                    log(correlation_scores)
                    log(agent)
                if self.increment_save:
                    self.visualizer.just_save(os.path.join(self.just_save_root, f"{sample_idx}_in_{agent}_{time.time_ns()}.png"))
                else:
                    self.visualizer.just_save(os.path.join(self.just_save_root, f"in_{agent}.png"))
                self.visualizer.clean()

        example_seq[present_idx][ego_dix]['data_samples'].gt_instances_3d['correlation'] = \
        np.ones_like(example_seq[present_idx][ego_dix]['data_samples'].gt_instances_3d.track_id, dtype=np.float32) # type: ignore
        input_dict['example_seq'] = example_seq

        return input_dict


@TRANSFORMS.register_module()
class GatherHistoryPoint(BaseTransform):
    def __init__(self, pad_delay: bool = True) -> None:
        self.pad_delay = pad_delay
    def transform(self, input_dict: Dict) -> Union[Dict,Tuple[List, List],None]:
        sample_interval = input_dict['sample_interval'] # 序列采集间隔
        present_idx = input_dict['present_idx'] # 当前序列第几位开始是关键帧, 比如0
        seq_length = input_dict['seq_length'] # 当前序列长度，比如6
        co_agents = copy.deepcopy(input_dict['co_agents']) # 参与协同的代理名称
        example_seq = input_dict['example_seq'] # 经过单车pipline得到的输入和标签数据
        history_seq_position = range(seq_length)[:present_idx+1] # 历史帧索引
        history_length = len(history_seq_position) # 历史长度
        delays = [-i * sample_interval * 0.1 for i in range(history_length)][::-1] # 历史时间差
        future_loc_matrix = input_dict['loc_matrix'][:present_idx+1, ...] # c x 4 4 # 历史ego的实际位置 比如 3, 2, 4, 4

        after_dim = 4
        for j, agent in enumerate(co_agents):
            all_points = []
            for i, idx in enumerate(history_seq_position):
                # import pdb
                # pdb.set_trace()
                points = example_seq[idx][j]['inputs']['points']
                if self.pad_delay:
                    points = torch.cat([
                        points,
                        torch.full_like(points[..., :1], delays[i])
                    ], dim=-1)
                points = LiDARPoints(
                    points,
                    points_dim=points.shape[-1],
                )
                if i < history_length - 1:
                    trans = calc_relative_pose(future_loc_matrix[-1], future_loc_matrix[i])
                    all_points.append(points.convert_to(Coord3DMode.LIDAR, trans))
                else:
                    all_points.append(points)
            all_points = LiDARPoints.cat(all_points)
            example_seq[present_idx][j]['inputs']['points'] = all_points.tensor
        return input_dict


@TRANSFORMS.register_module()
class DestoryEGOBox(BaseTransform):
    def __init__(self, ego_id: int = -100) -> None:
        self.ego_id = ego_id

    def transform(self, input_dict: Dict) -> Union[Dict,Tuple[List, List],None]:
        seq_length = input_dict['seq_length']
        co_length = input_dict['co_length']
        example_seq = input_dict['example_seq']
        for i in range(seq_length):
            for j in range(co_length):
                valid_mask = example_seq[i][j]['data_samples'].gt_instances_3d.track_id != self.ego_id
                example_seq[i][j]['data_samples'].gt_instances_3d = example_seq[i][j]['data_samples'].gt_instances_3d[valid_mask]

        input_dict['example_seq'] = example_seq
        return input_dict


@TRANSFORMS.register_module()
class RemoveHistoryLabels(BaseTransform):
    def transform(self, input_dict: Dict) -> Union[Dict,Tuple[List, List],None]:
        present_idx = input_dict['present_idx']
        seq_length = input_dict['seq_length']
        co_length = input_dict['co_length']
        example_seq = input_dict['example_seq']
        history_seq_timestamps = range(seq_length)[:present_idx]
        for i in history_seq_timestamps:
            for j in range(co_length):
                example_seq[i][j]['data_samples'] = Det3DDataSample()
        return input_dict


@TRANSFORMS.register_module()
class RemoveFutureLabels(BaseTransform):
    def transform(self, input_dict: Dict) -> Union[Dict,Tuple[List, List],None]:
        present_idx = input_dict['present_idx']
        seq_length = input_dict['seq_length']
        co_length = input_dict['co_length']
        example_seq = input_dict['example_seq']
        future_seq_timestamps = range(seq_length)[present_idx+1:]
        for i in future_seq_timestamps:
            for j in range(co_length):
                example_seq[i][j]['data_samples'] = Det3DDataSample()
        return input_dict
    

@TRANSFORMS.register_module()
class RemoveHistoryInputs(BaseTransform):
    def transform(self, input_dict: Dict) -> Union[Dict,Tuple[List, List],None]:
        present_idx = input_dict['present_idx']
        seq_length = input_dict['seq_length']
        co_length = input_dict['co_length']
        example_seq = input_dict['example_seq']
        history_seq_timestamps = range(seq_length)[:present_idx]
        for i in history_seq_timestamps:
            for j in range(co_length):
                example_seq[i][j]['inputs'] = {}
        return input_dict


@TRANSFORMS.register_module()
class RemoveFutureInputs(BaseTransform):
    def transform(self, input_dict: Dict) -> Union[Dict,Tuple[List, List],None]:
        present_idx = input_dict['present_idx']
        seq_length = input_dict['seq_length']
        co_length = input_dict['co_length']
        example_seq = input_dict['example_seq']
        future_seq_timestamps = range(seq_length)[present_idx+1:]
        for i in future_seq_timestamps:
            for j in range(co_length):
                example_seq[i][j]['inputs'] = {}
        return input_dict


@TRANSFORMS.register_module()
class PackSceneInfo(BaseTransform):
    def __init__(self,
                 meta_keys: tuple = (
                    'scene_name', 'seq_length', 'present_idx', 'co_agents', 'co_length', 'scene_length',
                    'scene_timestamps', 'sample_idx', 'seq_timestamps', 'pose_matrix', 'future_motion_matrix',
                    'loc_matrix', 'future_motion_rela_matrix'
                    ),
                delete_ori_key: bool = True,
                
    ) -> None:
        self.meta_keys = meta_keys
        self.delete_ori_key = delete_ori_key
    def transform(self, results: Dict) -> Union[Dict,Tuple[List, List],None]:
        scene_meta = BaseDataElement()
        meta_info = {}
        for key in self.meta_keys:
            if key in results:
                meta_info[key] = results[key]
                if self.delete_ori_key:
                    del results[key]
        scene_meta.set_metainfo(meta_info)
        results['scene_info'] = scene_meta
        return results


@TRANSFORMS.register_module()
class DropSceneKeys(BaseTransform):
    def __init__(self, keys:Tuple[str]) -> None:
        self.keys = keys
    def transform(self, results: Dict) -> Union[Dict,Tuple[List, List],None]:
        for key in self.keys:
            if key in results:
                del results[key]
        return results
        
