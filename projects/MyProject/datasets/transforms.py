from typing import List, Optional, Union,  Dict, Tuple
from numpy import ndarray
from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.loading import LoadPointsFromFile, LoadAnnotations3D
from mmcv.transforms.base import BaseTransform
import numpy as np
from mmengine.structures import BaseDataElement
from mmdet3d.structures import LiDARInstance3DBoxes, CameraInstance3DBoxes, DepthInstance3DBoxes, Det3DDataSample
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from os import path as osp
import copy
import torch
import cv2
from ..utils import calc_relative_pose, mat2vec
import matplotlib.pyplot as plt

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
                    'lidar2ego', 'bev_path', 'vehicle_speed_x', 'vehicle_speed_y',
                ),
                **kwargs) -> None:
        super().__init__(*args, meta_keys = meta_keys, **kwargs)
        

    def transform(self, results: Dict) -> Union[Dict,Tuple[List, List],None]:
        return super().transform(results) # type: ignore

### Scene Pipline ###

@TRANSFORMS.register_module()
class GatherV2XPoseInfo(BaseTransform):
    def transform(self, input_dict: Dict) -> Union[Dict,Tuple[List, List],None]:
        co_length = input_dict['co_length']
        seq_length = input_dict['seq_length']
        example_seq = input_dict['example_seq']

        seq_pose_matrix = []
        for i, data_info_list in enumerate(example_seq):
            co_pose_matrix = []
            for j in range(co_length):
                ego2global = np.array(data_info_list[j]['data_samples'].metainfo['ego2global'], dtype=np.float32)
                other2global = [np.array(data_info_list[k]['data_samples'].metainfo['ego2global'], dtype=np.float32) for k in range(co_length)]
                co_pose_matrix.append(calc_relative_pose(ego2global, other2global)) # type: ignore
            co_pose_matrix = np.stack(co_pose_matrix, axis=0)
            seq_pose_matrix.append(co_pose_matrix)
        seq_pose_matrix = np.stack(seq_pose_matrix, axis=0)
        input_dict['pose_matrix'] = seq_pose_matrix # seq, co, co, 4, 4

        motion_matrix = []
        for j in range(co_length):
            motion_list = [np.array(example_seq[i][j]['data_samples'].metainfo['ego2global'], dtype=np.float32) for i in range(seq_length)]
            base_motion = motion_list[0]
            motion_matrix.append(np.stack(calc_relative_pose(base_motion, motion_list), axis=0)) # type: ignore
        motion_matrix = np.stack(motion_matrix, axis=0) # NOTE ends with identity
        input_dict['motion_matrix'] = motion_matrix # co seq 4 4 
        # FIXME
        # FIXME NOT MOTION !
        
        return input_dict


@TRANSFORMS.register_module()
class ConvertMotionLabels(BaseTransform):
    def __init__(self,
                 pc_range,
                 voxel_size,
                 ego_id: int = -100,
                 ignore_index:int = 255,
                 filter_invalid: bool = True,
                 only_vehicle: bool = False,
                 vehicle_id_list: list = [0, 1, 2],
                 ) -> None:
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.ego_id = ego_id
        self.ignore_index = ignore_index
        self.filter_invalid = filter_invalid
        self.only_vehicle = only_vehicle
        self.vehicle_id_list = vehicle_id_list

        self.voxel_size = np.array(self.voxel_size).astype(np.float32)

        self.grid_size = np.array([
            np.ceil((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1]), # H
            np.ceil((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0]), # W
            np.ceil((self.pc_range[5] - self.pc_range[2]) / self.voxel_size[2]), # D
        ]).astype(np.int32)

        self.offset_xy = np.array([
            self.pc_range[0] + self.voxel_size[0] * 0.5,
            self.pc_range[1] + self.voxel_size[1] * 0.5
        ]).astype(np.float32)

    def transform(self, input_dict: Dict) -> Union[Dict,Tuple[List, List],None]:
        present_idx = input_dict['present_idx']
        seq_length = input_dict['seq_length']
        co_agents = input_dict['co_agents']
        example_seq = input_dict['example_seq']
        future_seq_timestamps = range(seq_length)[present_idx:]
        future_motion_matrix = input_dict['motion_matrix'][:, present_idx:] # c x 4 4

        track_instance_map = {}
        for j, agent in enumerate(co_agents):
            segmentations = []
            instances = []
            for i in future_seq_timestamps:

                data_samples = example_seq[i][j]['data_samples']

                gt_instances_3d = data_samples.gt_instances_3d.clone() # agent -> seq -> data_info

                segmentation = np.zeros((self.grid_size[0], self.grid_size[1]))
                instance = copy.deepcopy(segmentation)

                if self.only_vehicle:
                    vehicle_mask = np.isin(gt_instances_3d.labels_3d, self.vehicle_id_list)
                    gt_instances_3d = gt_instances_3d[vehicle_mask]

                lidar2ego = np.array(data_samples.metainfo['lidar2ego'], dtype=torch.float32) # type: ignore
                gt_instances_3d.bboxes_3d.rotate(lidar2ego[:3, :3].T, None)
                gt_instances_3d.bboxes_3d.translate(lidar2ego[:3, 3])

                if agent != 'infrastructure':
                    ego_instance = gt_instances_3d[gt_instances_3d.track_id == self.ego_id]
                    ego_bev_corners = ego_instance.bboxes_3d.corners[0, [0, 3, 7, 4], :2].numpy() # cpu tensor -> numpy corners N 8 3， clockwise
                    # [1, 4, 2]
                    ego_bev_corners_voxel = np.round((ego_bev_corners + self.offset_xy) / self.voxel_size[:2]).astype(np.int32)

                    if self.ego_id not in track_instance_map:
                        track_instance_map[self.ego_id] = len(track_instance_map) + 1
                    color = track_instance_map[self.ego_id] # color

                    poly_region = ego_bev_corners_voxel[0]

                    cv2.fillPoly(segmentation, [poly_region], 1.0) # type: ignore seg
                    cv2.fillPoly(instance, [poly_region], color) # instance seg

                if self.filter_invalid:
                    gt_instances_3d = gt_instances_3d[gt_instances_3d.bbox_3d_isvalid]

                if len(gt_instances_3d) > 0:
                    bev_corners = gt_instances_3d.bboxes_3d.corners[0, [0, 3, 7, 4], :2].numpy() # cpu tensor -> numpy corners N 8 3， clockwise
                    # [N, 4, 2]
                    bev_corners_voxel = np.round((bev_corners + self.offset_xy) / self.voxel_size[:2]).astype(np.int32)

                    for idx, id in enumerate(gt_instances_3d.track_id):
                        if id not in track_instance_map:
                            track_instance_map[id] = len(track_instance_map) + 1
                        color = track_instance_map[id]

                        poly_region = bev_corners_voxel[idx]

                        cv2.fillPoly(segmentation, [poly_region], 1.0) # type: ignore seg
                        cv2.fillPoly(instance, [poly_region], color) # instance seg
                
                segmentations.append(segmentation.astype(np.int64)) # type: ignore
                instances.append(instance.astype(np.int64)) # type: ignore
            segmentations = np.stack(segmentations, axis=0) # xx h w
            instances = np.stack(instances, axis=0) # xx h w

            # FIXME
            # FIXME
            # FIXME
            # FIXME
            # FIXME

            future_motion = mat2vec(future_motion_matrix[j, ...])








        return None


@TRANSFORMS.register_module()
class ImportanceFilter(BaseTransform):
    def __init__(self,
                 pc_range,
                 voxel_size,
                 ego_id: int = -100,
                 ego_name: str = 'ego_vehicle',
                 only_vehicle: bool = False,
                 vehicle_id_list: list = [0, 1, 2],
                 ignore_thres: float = 0.5,
                 interrrupt_thres: float = 10,
                 visualize: Optional[str] = None,
                 ) -> None:
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.ego_id = ego_id
        self.ego_name = ego_name
        self.only_vehicle = only_vehicle
        self.vehicle_id_list = vehicle_id_list
        self.ignore_thres = ignore_thres
        self.interrrupt_thres = interrrupt_thres
        self.visualize = visualize

        self.voxel_size = np.array(self.voxel_size).astype(np.float32)

        self.grid_size = np.array([
            np.ceil((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1]), # H
            np.ceil((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0]), # W
            np.ceil((self.pc_range[5] - self.pc_range[2]) / self.voxel_size[2]), # D
        ]).astype(np.int32) # 1024 1024 1

        self.offset_xy = np.array([
            self.pc_range[0] + self.voxel_size[0] * 0.5,
            self.pc_range[1] + self.voxel_size[1] * 0.5
        ]).astype(np.float32)

        if self.visualize:
            self.scatter_trans = np.array(
                [[0, 1],
                [-1, 0]],
                dtype=np.float32
            )
            self.fig, self.ax = plt.subplots(1, 1)

        self.cmaps = [
            'Oranges', 'Greens', 'Purples', 'Oranges', 'PuRd', 'BuPu',
        ]
        # self.cmaps = [
        #     'Greys', 'Purples', 'Greens', 'Oranges', 'Reds',
        #     'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        #     'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
        # ]

    def transform(self, input_dict: Dict) -> Union[Dict,Tuple[List, List],None]:
        present_idx = input_dict['present_idx']
        seq_length = input_dict['seq_length']
        co_agents = copy.deepcopy(input_dict['co_agents'])
        example_seq = input_dict['example_seq']
        future_seq_timestamps = range(seq_length)[present_idx:]
        future_length = len(future_seq_timestamps)
        future_pose_matrix = input_dict['pose_matrix'][present_idx:, ...] # x c c 4 4
        future_motion_matrix = input_dict['motion_matrix'][:, present_idx:] # c x 4 4

        co_agents = {v : k for k, v in enumerate(co_agents)}

        assert self.ego_name in co_agents and self.ego_name != 'infrastructure'
        track_map_list = []
        for agent, j in co_agents.items():
            if agent == self.ego_name:
                ego_motion_matrix = future_motion_matrix[j] # x 4 4 rela to present
                persent_trans_matrix = future_pose_matrix[0, :, j]
                persent_trans_matrix = np.delete(persent_trans_matrix, j, axis=0) # ego to [ego, other, ...] at present c-1 4 4
            else:
                track_map = {}
                agent_motion_matrix = future_motion_matrix[j] # x 4 4 rela to present
                for i, timestamp in enumerate(future_seq_timestamps):
                    gt_instances_3d = example_seq[timestamp][j]['data_samples'].gt_instances_3d.clone()

                    if self.only_vehicle:
                        vehicle_mask = np.isin(gt_instances_3d.labels_3d, self.vehicle_id_list)
                        gt_instances_3d = gt_instances_3d[vehicle_mask]
                    
                    if agent != 'infrastructure':
                        ego_mask = gt_instances_3d.track_id != self.ego_id
                        gt_instances_3d = gt_instances_3d[ego_mask]

                    gt_instances_3d = gt_instances_3d[gt_instances_3d.bbox_3d_isvalid] # only visible target

                    lidar2agent = np.array(example_seq[timestamp][j]['data_samples'].metainfo['lidar2ego'], dtype=np.float32) # type: ignore
                    agent2present = agent_motion_matrix[0]
                    lidar2present = lidar2agent @ agent2present
                    gt_instances_3d.bboxes_3d.rotate(lidar2present[:3, :3].T, None)
                    gt_instances_3d.bboxes_3d.translate(lidar2present[:3, 3])
                    
                    for idx, id in enumerate(gt_instances_3d.track_id):
                        bboxes_xy = gt_instances_3d.bboxes_3d.tensor[idx].numpy()[:2] # agent坐标系下其他目标的present的中心
                        labels = gt_instances_3d.labels_3d[idx]
                        if id not in track_map:
                            track_map[id] = {
                                # 'color': self.cmaps[np.random.randint(0, len(self.cmaps))],
                                'color': self.cmaps[labels],
                                'center': [],
                                'start': i
                            }
                        track_map[id]['center'].append(bboxes_xy)
                track_map_list.append(track_map)

        ego_dix = co_agents.pop(self.ego_name) # ignore the ego

        for i, (agent, j) in enumerate(co_agents.items()): # agent
            track_map = track_map_list[i]
            present_instances_3d = example_seq[present_idx][j]['data_samples'].gt_instances_3d
            trans = persent_trans_matrix[i] # type: ignore
            rela_matrix = np.stack([trans @ ego_motion_matrix[k] for k in range(future_length)], axis=0) # type: ignore
            rela_centers = rela_matrix[:, :2, 3]
            
            if self.visualize:
                # rela_centers_vis = rela_centers @ self.scatter_trans.T # type: ignore
                # FIXME 这里其实是BEV分割图片的问题，按照原方式编码才能对齐点云特征
                # rela_centers_vis = rela_centers
                rela_centers_voxel = np.round((rela_centers - self.offset_xy) / self.voxel_size[:2]).astype(np.int32)
                self.ax.imshow(np.full((self.grid_size[0], self.grid_size[1], 3), 127, dtype=np.uint8)) # type: ignore
                self.ax.scatter(rela_centers_voxel[:, 0], rela_centers_voxel[:, 1], c = np.linspace(0.0, 1.0, future_length)[::-1], cmap='Blues', s=8) # type: ignore
            
            valid_id = []
            for id, v in track_map.items():
                color = v['color']
                start = v['start']
                centers = np.stack(v['center'], axis=0)
                cmp_centers = rela_centers[start:start + len(centers)]
                distance = np.linalg.norm(centers - cmp_centers, axis=1)
                interrupt_mask = np.logical_and(distance < self.interrrupt_thres, distance > self.ignore_thres)
                isvalid = np.any(interrupt_mask)
                if isvalid:
                    valid_id.append(id)

                if self.visualize:
                    # centers_vis = centers @ self.scatter_trans.T # type: ignore
                    # FIXME 这里其实是BEV分割图片的问题，按照原方式编码才能对齐点云特征
                    # centers_vis = centers
                    voxel_centers = np.round((centers - self.offset_xy) / self.voxel_size[:2]).astype(np.int32)
                    if isvalid:
                        self.ax.scatter(voxel_centers[: ,0], voxel_centers[:, 1], c = np.linspace(0.0, 1.0, len(voxel_centers))[::-1], cmap=color, s=3) # type: ignore
                    else:
                        self.ax.scatter(voxel_centers[: ,0], voxel_centers[:, 1], c = np.linspace(0.0, 1.0, len(voxel_centers))[::-1], cmap='binary', s=1) # type: ignore
            
            valid_id_mask = np.isin(present_instances_3d.track_id, valid_id)
            
            present_instances_3d['importance'] = valid_id_mask
            example_seq[present_idx][j]['data_samples'].gt_instances_3d = present_instances_3d
            if self.visualize:
                print(valid_id)
                print(agent)
                print(present_instances_3d.track_id)
                print(valid_id_mask)
                self.fig.savefig(self.visualize, dpi = 300) # type: ignore # FIXME MORE FIG ONCE
                self.ax.cla()

        example_seq[present_idx][ego_dix]['data_samples'].gt_instances_3d['importance'] = \
        np.full_like(example_seq[present_idx][ego_dix]['data_samples'].gt_instances_3d.bbox_3d_isvalid, True) # type: ignore
        input_dict['example_seq'] = example_seq

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
class PackSceneInfo(BaseTransform):
    def __init__(self,
                 meta_keys: tuple = (
                    'scene_name', 'seq_length', 'present_idx', 'co_agents', 'co_length', 'scene_length',
                    'scene_timestamps', 'sample_idx', 'seq_timestamps', 'pose_matrix', 'motion_matrix'
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
        
