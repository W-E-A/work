from typing import List, Optional, Union,  Dict, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from numpy import ndarray
from mmdet3d.registry import TRANSFORMS, VISUALIZERS
from mmdet3d.datasets.transforms.loading import LoadPointsFromFile, LoadAnnotations3D
from mmcv.transforms.base import BaseTransform
from mmengine.structures import BaseDataElement
from mmdet3d.structures import LiDARInstance3DBoxes, CameraInstance3DBoxes, DepthInstance3DBoxes, Det3DDataSample, LiDARPoints, Coord3DMode
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from os import path as osp
import copy
import cv2
import matplotlib.pyplot as plt
import torch
import os
from mmengine.logging import print_log
import logging
def log(msg = "" ,level: int = logging.INFO):
    print_log(msg, "current", level)
from ..utils import calc_relative_pose, convert_instance_mask_to_center_and_offset_label

### Transform Template ###

# @TRANSFORMS.register_module()
# class NoOps(BaseTransform):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#     def transform(self, input_dict: Dict) -> Union[Dict,Tuple[List, List],None]:
#         return input_dict


### Single Agent Pipline ###

@TRANSFORMS.register_module()
class LoadPointsNPZ(LoadPointsFromFile):
    def __init__(self, *args, **kwargs) -> None:
        super(LoadPointsNPZ, self).__init__(*args, **kwargs)

    def _load_points(self, pts_filename: str) -> ndarray:
        assert osp.exists(pts_filename) and pts_filename.endswith('.npz')
        pc = np.load(pts_filename)['data']
        return pc


@TRANSFORMS.register_module()
class LoadPointsFromMultiSweepsNPZ(BaseTransform):
    def __init__(self,
        use_multi_sweeps: bool = False,
        remove_point_cloud_range: Optional[List[float]] = None,
        pad_empty_sweeps: int = 0,
        del_sweeps: bool = True,
        *args,
        **kwargs,) -> None:
        self.loader = LoadPointsNPZ(*args, **kwargs)
        self.use_multi_sweeps = use_multi_sweeps
        self.remove_point_cloud_range = remove_point_cloud_range
        self.pad_empty_sweeps = pad_empty_sweeps
        self.del_sweeps = del_sweeps

    def _remove_inner_range(self, points, remove_point_cloud_range):
        points_mask = points.in_range_3d(remove_point_cloud_range)
        points_mask = torch.logical_not(points_mask)
        return points[points_mask]

    def transform(self, results: dict) -> dict:
        assert 'points' in results.keys()
        points = results['points'] # BasePoints
        if self.use_multi_sweeps:
            dtype = points.tensor.dtype
            device = points.tensor.device
            points.tensor = torch.cat([points.tensor, torch.full((points.tensor.shape[0], 1), 0.0).to(dtype).to(device)], dim=1)
            points.points_dim = points.tensor.shape[-1]
        if self.remove_point_cloud_range:
            points = self._remove_inner_range(points, self.remove_point_cloud_range)
        sweep_points_list = [points]
        if self.use_multi_sweeps:
            ts = results['timestamp']
            if 'lidar_sweeps' not in results:
                if self.pad_empty_sweeps > 0:
                    for i in range(self.pad_empty_sweeps):
                        sweep_points_list.append(points)
            else:
                for sweep in results['lidar_sweeps']:
                    sweep_ts = sweep['timestamp']
                    sweep2key = sweep['sweep2key']
                    points_sweep = self.loader.transform(sweep)['points']
                    dtype = points_sweep.tensor.dtype
                    device = points_sweep.tensor.device
                    points_sweep.tensor = torch.cat([points_sweep.tensor, torch.full((points_sweep.tensor.shape[0], 1), 0.1 * (ts - sweep_ts)).to(dtype).to(device)], dim=1)
                    points_sweep.points_dim = points_sweep.tensor.shape[-1]
                    if self.remove_point_cloud_range:
                        points_sweep = self._remove_inner_range(points_sweep, self.remove_point_cloud_range)
                    sweep_points_list.append(points_sweep.convert_to(Coord3DMode.LIDAR, sweep2key))
            if self.del_sweeps and 'lidar_sweeps' in results:
                results.pop('lidar_sweeps')
        points = points.cat(sweep_points_list)
        results['points'] = points
        return results


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
    def __init__(self, ego_id: int = -100, infrastructure_name: str = 'infrastructure') -> None:
        self.ego_id = ego_id
        self.infrastructure_name = infrastructure_name

    def transform(self, input_dict: Dict) -> Union[Dict,Tuple[List, List],None]:
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        track_id = input_dict['track_id']
        ego_box = gt_bboxes_3d.tensor[track_id == self.ego_id]
        if ego_box.shape[0] != 1 and input_dict['agent'] != self.infrastructure_name:
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
class CorrelationFilter(BaseTransform):
    def __init__(self,
        pc_range,
        voxel_size,
        infrastructure_name: str = 'infrastructure',
        with_velocity: bool = True,
        ego_id: int = -100,
        min_distance_thres: float = 5,
        max_distance_thres: float = 20,
        alpha_coeff: float = 1,
        beta_coeff: float = 1,
        gamma_coeff: float = 2,
        enable_visualize: bool = False,
        visualizer_cfg: dict = None,
        just_save_root: str = None,
        increment_save: bool = True,
        verbose: bool = False
    ) -> None:
        self.infrastructure_name = infrastructure_name
        self.with_velocity = with_velocity
        self.ego_id = ego_id
        self.min_distance_thres = min_distance_thres
        self.max_distance_thres = max_distance_thres
        self.alpha_coeff = alpha_coeff
        self.beta_coeff = beta_coeff
        self.gamma_coeff = gamma_coeff
        self.enable_visualize = enable_visualize
        if self.enable_visualize:
            visualizer_cfg = dict(
                type='SimpleLocalVisualizer',
                pc_range=pc_range,
                voxel_size=voxel_size,
                name='correlation_visualizer',
            ) if not visualizer_cfg else visualizer_cfg
            self.visualizer: SimpleLocalVisualizer = VISUALIZERS.build(visualizer_cfg)
            self.just_save_root = './data/vis/instance_correlation_gt' if not just_save_root else just_save_root
            self.increment_save = increment_save
        else:
            self.visualizer = None
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
        rela_vector_norm = np.clip(rela_vector_norm, 1e-1, np.inf)
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
        co_agents = input_dict['co_agents'] # 参与协同的代理名称
        example_seq = input_dict['example_seq'] # 经过单车pipline得到的输入和标签数据
        future_seq_position = range(seq_length)[present_idx:] # 取出未来帧在序列中的位置， 比如3, 4, 5
        future_length = len(future_seq_position) # 表明未来帧的长度，比如3
        future_pose_matrix = input_dict['pose_matrix'][present_idx:, ...] # x c c 4 4 # 未来帧的每帧变换关系
        future_motion_matrix = input_dict['future_motion_matrix'] # c x 4 4 未来帧相对于起始帧的位姿关系
        future_loc_matrix = input_dict['loc_matrix'][present_idx:, :, ...] # x c 4 4 # 未来帧ego的实际位置

        if self.verbose:
            log(scene_name)
            log(seq_timestamps)

        assert self.infrastructure_name in co_agents
        infrastructure_idx = co_agents.index(self.infrastructure_name)
        # 如果处理的是路侧，则对于路侧在时序上跟踪所有目标，相对于关键帧的轨迹，路侧得到一个track_map
        track_map = {}
        for i, timestamp in enumerate(future_seq_position):
            gt_instances_3d = example_seq[timestamp][infrastructure_idx]['data_samples'].gt_instances_3d.clone()

            # if self.only_vehicle:
            #     vehicle_mask = np.isin(gt_instances_3d.labels_3d, self.vehicle_id_list)
            #     gt_instances_3d = gt_instances_3d[vehicle_mask]

            # gt_instances_3d = gt_instances_3d[gt_instances_3d.bbox_3d_isvalid] # only visible target bug do not modify

            lidar2agent = np.array(example_seq[timestamp][infrastructure_idx]['data_samples'].metainfo['lidar2ego'], dtype=np.float32) # type: ignore
            agent2present = future_motion_matrix[infrastructure_idx][i]
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

        correlation_list = []
        for j, agent in enumerate(co_agents):
            if agent != self.infrastructure_name: # 除了infrastructure之外的每个车分别都是ego
                # 如果处理的是ego则先得到ego的规划轨迹和关键帧相对于其他协同对象的变换矩阵
                ego_motion_matrix = future_motion_matrix[j] # x 4 4 rela to present
                trans = future_pose_matrix[0, infrastructure_idx, j] # c 4 4
                if self.with_velocity:
                    ego_present_vel = np.array(
                        [example_seq[present_idx][j]['data_samples'].metainfo['vehicle_speed_x'],
                         example_seq[present_idx][j]['data_samples'].metainfo['vehicle_speed_y']],
                         dtype=np.float32
                    )
                    if self.verbose:
                        log(f"ego : {ego_present_vel}")

                # 对于每个协同代理的视角下，将ego的规划轨迹转到协同代理的当前坐标系下
                present_instances_3d = example_seq[present_idx][infrastructure_idx]['data_samples'].gt_instances_3d
                rela_matrix = np.stack([trans @ ego_motion_matrix[k] for k in range(future_length)], axis=0) # type: ignore
                rela_centers = rela_matrix[:, :2, 3] # x 2
                if self.with_velocity:
                    rots = np.linalg.inv(future_loc_matrix[0, infrastructure_idx, ...])[:2, :2]# 每个ego速度对于世界，所以转到对于present的代理描述下
                    rela_vel = ego_present_vel @ rots.T # 转置乘法

                if self.visualizer:
                    self.visualizer.set_points_from_npz(example_seq[present_idx][infrastructure_idx]['data_samples'].metainfo["lidar_path"])
                
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
                
                correlation_list.append(torch.tensor(correlation_scores, dtype=torch.float32))

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
                        save_path = os.path.join(self.just_save_root, f"{agent}")
                        self.visualizer.just_save(os.path.join(save_path, f"{scene_name}_{seq_timestamps[0]}.png"))
                    else:
                        save_path = os.path.join(self.just_save_root, f"{agent}")
                        self.visualizer.just_save(os.path.join(save_path, f"{scene_name}.png"))
                    self.visualizer.clean()
        
        correlation_list = torch.stack(correlation_list, dim=1) # N, x
        example_seq[present_idx][infrastructure_idx]['data_samples'].gt_instances_3d['correlations'] = correlation_list
        
        return input_dict


@TRANSFORMS.register_module()
class MakeMotionLabels(BaseTransform):
    def __init__(self,
        pc_range,
        voxel_size,
        infrastructure_name: str = 'infrastructure',
        mode: str = 'normal',
        just_present: bool = False,
        ego_id: int = -100,
        only_vehicle: bool = False,
        vehicle_id_list: list = [0, 1, 2],
        filter_invalid: bool = True,
        ignore_index:int = 255,
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

        self.mode = mode
        assert self.mode in ('normal', 'ego', 'inf', 'corr')
        # normal 返回所有目标的motion
        # ego 返回所有ego自己在inf坐标系的motion
        # inf 仅返回inf的所有目标的motion
        # corr 必须先运行CorrelationFilter使得infra的每个目标有个打分
        self.just_present = just_present
        self.infrastructure_name = infrastructure_name
        self.ego_id = ego_id
        self.only_vehicle = only_vehicle
        self.vehicle_id_list = vehicle_id_list
        self.filter_invalid = filter_invalid
        self.ignore_index = ignore_index

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
        # future_motion_matrix = input_dict['future_motion_matrix'] # c x 4 4 未来帧相对于起始帧的位姿关系
        # future_loc_matrix = input_dict['loc_matrix'][present_idx:, :, ...] # c x 4 4 # 未来帧ego的实际位置
        future_motion_rela_matrix = input_dict['future_motion_rela_matrix'] # c x 4 4 未来帧对于前一帧的位姿关系

        assert self.infrastructure_name in co_agents
        infrastructure_idx = co_agents.index(self.infrastructure_name)
        inf_motion_rela_matrix = future_motion_rela_matrix[infrastructure_idx]

        if self.mode == 'corr':
            corr_heatmaps = []
            corr_gt_masks = []
            corr_dilate_heatmaps = []
            gt_instances_3d = example_seq[present_idx][infrastructure_idx]['data_samples'].gt_instances_3d.clone()
            assert 'correlations' in gt_instances_3d.keys()
            if self.only_vehicle:
                vehicle_mask = np.isin(gt_instances_3d.labels_3d, self.vehicle_id_list)
                gt_instances_3d = gt_instances_3d[vehicle_mask]
            # if self.filter_invalid: # do not modify
            #     gt_instances_3d = gt_instances_3d[gt_instances_3d.bbox_3d_isvalid] # do not modify
            bbox_corners = gt_instances_3d.bboxes_3d.corners[:, [0, 3, 7, 4], :2].numpy() # four corner B, 4, 2
            bbox_corners_voxel = np.round((bbox_corners - self.offset_xy) / self.voxel_size[:2]).astype(np.int32) # N 4 2 to voxel coor
            correlations = gt_instances_3d.correlations
            for j in range(len(co_agents) - 1):
                correlation = correlations[:, j]
                corr_heatmap = np.zeros((self.grid_size[0], self.grid_size[1]))
                for index, id in enumerate(gt_instances_3d.track_id):
                    score = correlation[index].item()
                    poly_region = bbox_corners_voxel[index]
                    cv2.fillPoly(corr_heatmap, [poly_region], score)
                kernel_size = 3
                # 使用膨胀核对图像进行膨胀，保持原分数位置不变
                corr_gt_mask = corr_heatmap > 0
                corr_heatmap = cv2.dilate(corr_heatmap, np.ones((kernel_size, kernel_size), dtype=np.uint8))
                corr_dilate_heatmaps.append(cv2.dilate(corr_heatmap, np.ones((kernel_size, kernel_size), dtype=np.uint8)))
                corr_heatmap = cv2.GaussianBlur(corr_heatmap, (kernel_size, kernel_size), 0)
                corr_heatmaps.append(corr_heatmap)
                corr_gt_masks.append(corr_gt_mask)
            corr_heatmaps = torch.from_numpy(np.stack(corr_heatmaps, axis=0)).float() # c-1, H, W
            corr_gt_masks = torch.from_numpy(np.stack(corr_gt_masks, axis=0)) # c-1, H, W
            corr_dilate_heatmaps = torch.from_numpy(np.stack(corr_dilate_heatmaps, axis=0)).float() # c-1, H, W
            input_dict['example_seq'][present_idx][infrastructure_idx]['corr_heatmaps'] = corr_heatmaps
            input_dict['example_seq'][present_idx][infrastructure_idx]['corr_gt_masks'] = corr_gt_masks
            input_dict['example_seq'][present_idx][infrastructure_idx]['corr_dilate_heatmaps'] = corr_dilate_heatmaps
        else:
            for j, agent in enumerate(co_agents):
                if self.mode == 'ego' and agent == self.infrastructure_name:
                    continue
                if self.mode == 'inf' and agent != self.infrastructure_name:
                    continue
                track_map = {}
                segmentations = []
                instances = []
                for i, timestamp in enumerate(future_seq_position):
                    if self.just_present and i != 0:
                        break
                    gt_instances_3d = example_seq[timestamp][j]['data_samples'].gt_instances_3d.clone()
                    if self.mode == 'ego':
                        ego_mask = gt_instances_3d.track_id == self.ego_id
                        gt_instances_3d = gt_instances_3d[ego_mask]
                        assert len(gt_instances_3d) == 1 # ego only, must construct the ego bbox

                    if gt_instances_3d.bboxes_3d is None:
                        segmentation = np.ones(
                            (self.grid_size[0], self.grid_size[1])) * self.ignore_index # H, W
                        instance = np.ones_like(segmentation) * self.ignore_index
                    else:
                        segmentation = np.zeros((self.grid_size[0], self.grid_size[1])) # H, W
                        instance = np.zeros_like(segmentation)    
                    if not self.mode == 'ego':
                        if self.only_vehicle:
                            vehicle_mask = np.isin(gt_instances_3d.labels_3d, self.vehicle_id_list)
                            gt_instances_3d = gt_instances_3d[vehicle_mask]
                        
                        if self.filter_invalid:
                            gt_instances_3d = gt_instances_3d[gt_instances_3d.bbox_3d_isvalid]

                    if self.mode == 'ego':
                        trans = future_pose_matrix[i, infrastructure_idx, j, ...] # 4, 4
                        gt_instances_3d.bboxes_3d.rotate(trans[:3, :3].T, None)
                        gt_instances_3d.bboxes_3d.translate(trans[:3, 3])
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
                    future_motion_rela_matrix[j] if not self.mode == 'ego' else inf_motion_rela_matrix,
                    len(track_map),
                    ignore_index = self.ignore_index,
                    spatial_extent = self.warp_size,
                ) # len, 1, h, w  len, 2, h, w  len, 2, h, w

                # 如果分割有任何是没有box的则清除中心度为ignore
                invalid_mask = (segmentations[:, 0, 0] == self.ignore_index)
                instance_centerness[invalid_mask] = self.ignore_index
                motion_label = {
                    'motion_segmentation': torch.from_numpy(segmentations).float(), # [fu_len, H, W]
                    'motion_instance': torch.from_numpy(instances).float(), # [fu_len, H, W]
                    'instance_centerness': torch.from_numpy(instance_centerness).float(), # len, 1, h, w
                    'instance_offset': torch.from_numpy(instance_offset).float(), # len, 2, h, w
                    'instance_flow': torch.from_numpy(instance_flow).float(), # len, 2, h, w invalid at last
                    'future_egomotion': torch.from_numpy(future_motion_rela_matrix[j] if not self.mode == 'ego' else inf_motion_rela_matrix), # len, 4, 4 ident at 0
                }
                if self.mode == 'ego':
                    input_dict['example_seq'][present_idx][j]['ego_motion_label'] = motion_label
                elif self.mode == 'inf':
                    input_dict['example_seq'][present_idx][j]['inf_motion_label'] = motion_label
                else:
                    input_dict['example_seq'][present_idx][j]['motion_label'] = motion_label
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
        
