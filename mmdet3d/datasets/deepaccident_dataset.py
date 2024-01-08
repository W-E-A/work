from typing import Callable, List, Optional, Sequence, Union,  Dict, Tuple, Mapping, Any
from numpy import ndarray
from mmdet3d.registry import DATASETS, TRANSFORMS
from .det3d_dataset import Det3DDataset
from .transforms.loading import LoadPointsFromFile
from mmcv.transforms.base import BaseTransform
from mmengine.dataset import BaseDataset, Compose
from mmengine.config import Config
import mmengine
from mmengine.logging import print_log
from logging import WARNING
import numpy as np
from mmengine.structures import BaseDataElement
from mmdet3d.structures import LiDARInstance3DBoxes, CameraInstance3DBoxes, DepthInstance3DBoxes
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.datasets.transforms.loading import LoadAnnotations3D
from os import path as osp
import copy

# @TRANSFORMS.register_module()
# class AddPoseNoise(BaseTransform):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#     def transform(self, results: Dict) -> Union[Dict,Tuple[List, List],None]:
#         return None


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
class ObjectTrackIDFilterV2X(BaseTransform):
    def __init__(self, ids: List[str]) -> None:
        self.ids = ids

    def transform(self, input_dict: dict) -> dict:

        track_id = input_dict['track_id']
        gt_bboxes_mask = np.array([id not in self.ids for id in track_id], dtype=bool)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]
        input_dict['bbox_3d_isvalid'] = input_dict['bbox_3d_isvalid'][gt_bboxes_mask]
        input_dict['track_id'] = input_dict['track_id'][gt_bboxes_mask]

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(ids={self.ids})'
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
        # packed_results = super().transform(results)
        # coop_samples = packed_results['data_samples'].clone() # type: ignore
        # single_samples = packed_results['data_samples'].clone() # type: ignore
        # if len(packed_results['data_samples'].gt_instances_3d) > 0: # type: ignore
        #     mask = packed_results['data_samples'].gt_instances_3d.bboxes_3d.tensor[:, -1] > 0.5 # type: ignore
        #     single_samples.gt_instances_3d = (single_samples.gt_instances_3d)[mask]
        #     coop_samples.gt_instances_3d.bboxes_3d.tensor = coop_samples.gt_instances_3d.bboxes_3d.tensor[:, :-1]
        #     coop_samples.gt_instances_3d.bboxes_3d.box_dim = coop_samples.gt_instances_3d.bboxes_3d.box_dim - 1
        #     single_samples.gt_instances_3d.bboxes_3d.tensor = single_samples.gt_instances_3d.bboxes_3d.tensor[:, :-1]
        #     single_samples.gt_instances_3d.bboxes_3d.box_dim = single_samples.gt_instances_3d.bboxes_3d.box_dim - 1
        # del packed_results['data_samples'] # type: ignore
        # packed_results['coop_samples'] = coop_samples # type: ignore
        # packed_results['single_samples'] = single_samples # type: ignore
        return super().transform(results) # type: ignore


@TRANSFORMS.register_module()
class PackSceneInfo(BaseTransform):
    def __init__(self,
                 meta_keys: tuple = (
                    'scene_name', 'seq_length', 'present_idx', 'co_agents', 'co_length', 'scene_length',
                    'scene_timestamps', 'sample_idx', 'seq_timestamps',
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
        

@DATASETS.register_module()
class DeepAccident_Dataset(BaseDataset):
    """
    metainfo: classes agents

    data_list: list of dict

        dict_keys([
            'scenario_meta', 'scenario_type', 'vehicle_name', 'scene_name', 
            'lidar_prefix', 'lidar_path', 'bev_path', 'timestamp', 'scenario_length', 
            'cams', 'lidar_to_ego_matrix', 'ego_to_world_matrix', 'vehicle_speed_x', 
            'vehicle_speed_y', 'gt_names', 'gt_boxes', 'gt_velocity', 'vehicle_id', 
            'num_lidar_pts', 'camera_visibility', 'sample_interval'])
        <class 'dict'>
        <class 'str'>
        <class 'str'>
        <class 'str'>
        <class 'str'>
        <class 'str'>
        <class 'str'>
        <class 'int'>
        <class 'int'>
        <class 'dict'>
        <class 'numpy.ndarray'>
        <class 'numpy.ndarray'>
        <class 'float'>
        <class 'float'>
        <class 'numpy.ndarray'>
        <class 'numpy.ndarray'>
        <class 'numpy.ndarray'>
        <class 'numpy.ndarray'>
        <class 'numpy.ndarray'>
        <class 'numpy.ndarray'>
        <class 'int'>

    self.data_list : agents scenario_frame sample_idx

    """
    METAINFO = {
        'classes': ('car', 'van', 'truck', 'cyclist', 'motorcycle', 'pedestrian'),
        'agents': ('ego_vehicle', 'other_vehicle', 'ego_vehicle_behind', 'other_vehicle_behind', 'infrastructure')
    }
    def __init__(self,
                 ann_file: str,
                 metainfo: Union[Mapping, Config, None] = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 **kwargs):

        super().__init__(ann_file,
                         metainfo,
                         data_prefix=dict(),
                         pipeline=pipeline, # type: ignore
                         test_mode=test_mode,
                         **kwargs)
        
    def load_data_list(self) -> List[dict]:
        """
        dict_keys(['name', 'metainfo', 'seq_timestamps', 'seq_length', 'agents', 'seq_data', 'sample_idx'])

        dict_keys(['ego_vehicle', 'other_vehicle', 'ego_vehicle_behind', 'other_vehicle_behind', 'infrastructure', 
        'gt_names', 'gt_boxes', 'gt_velocity', 'vehicle_id', 'num_lidar_pts', 'camera_visibility'])

        dict_keys(['lidar_prefix', 'lidar_path', 'bev_path', 'cams', 'lidar_to_ego_matrix', 'ego_to_world_matrix', 'vehicle_speed_x', 'vehicle_speed_y'])
        
        """
        annotations = mmengine.load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                            'keys')
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        total_scene_name = tuple(set([item['scene_name'] for item in raw_data_list]))

        scene_data = {scene_name : [] for scene_name in total_scene_name}
        for raw_data in raw_data_list:
            scene_data[raw_data['scene_name']].append(raw_data)


        data_list = []
        for scene_name, agent_data_list in scene_data.items():
            scene_dict = {}

            temp_data = agent_data_list[0]
            temp_name = temp_data['vehicle_name']
            timestamps = [frame['timestamp'] for frame in agent_data_list if frame['vehicle_name'] == temp_name]
            timestamps.sort()

            scene_dict['name'] = scene_name
            scene_dict['metainfo'] = temp_data['scenario_meta']
            scene_dict['seq_timestamps'] = timestamps
            scene_dict['seq_interval'] = temp_data['sample_interval']
            scene_dict['seq_length'] = len(timestamps)
            scene_dict['agents'] = self._metainfo['agents']
            scene_dict['seq_data'] = {timestamp: {agent : {} for agent in self._metainfo['agents']} for timestamp in timestamps}

            for frame in agent_data_list:
                agent = frame['vehicle_name']
                timestamp = frame['timestamp']

                scene_dict['seq_data'][timestamp][agent]['lidar_prefix'] = frame['lidar_prefix']
                scene_dict['seq_data'][timestamp][agent]['lidar_path'] = frame['lidar_path']
                scene_dict['seq_data'][timestamp][agent]['bev_path'] = frame['bev_path']
                scene_dict['seq_data'][timestamp][agent]['cams'] = frame['cams']
                scene_dict['seq_data'][timestamp][agent]['lidar_to_ego_matrix'] = frame['lidar_to_ego_matrix']
                scene_dict['seq_data'][timestamp][agent]['ego_to_world_matrix'] = frame['ego_to_world_matrix']
                scene_dict['seq_data'][timestamp][agent]['vehicle_speed_x'] = frame['vehicle_speed_x']
                scene_dict['seq_data'][timestamp][agent]['vehicle_speed_y'] = frame['vehicle_speed_y']

                if 'gt_names' not in scene_dict['seq_data'][timestamp].keys():
                    scene_dict['seq_data'][timestamp]['gt_names'] = frame['gt_names']
                if 'gt_boxes' not in scene_dict['seq_data'][timestamp].keys():
                    scene_dict['seq_data'][timestamp]['gt_boxes'] = frame['gt_boxes']
                if 'gt_velocity' not in scene_dict['seq_data'][timestamp].keys():
                    scene_dict['seq_data'][timestamp]['gt_velocity'] = frame['gt_velocity']
                if 'vehicle_id' not in scene_dict['seq_data'][timestamp].keys():
                    scene_dict['seq_data'][timestamp]['vehicle_id'] = frame['vehicle_id']
                if 'num_lidar_pts' not in scene_dict['seq_data'][timestamp].keys():
                    scene_dict['seq_data'][timestamp]['num_lidar_pts'] = frame['num_lidar_pts']
                if 'camera_visibility' not in scene_dict['seq_data'][timestamp].keys():
                    scene_dict['seq_data'][timestamp]['camera_visibility'] = frame['camera_visibility']

            data_list.append(scene_dict)
        return data_list


@DATASETS.register_module()
class DeepAccident_V2X_Dataset(Det3DDataset):
    """
    dict_keys(['seq', 'example_seq', 'scene_name', 'seq_length', 'present_idx',
    'co_agents', 'scene_length', 'scene_timestamps', 'sample_idx',
    'seq_timestamps', 'co_length',
    ])
    
    """
    METAINFO = {
        'classes': ('car', 'van', 'truck', 'cyclist', 'motorcycle', 'pedestrian'),
        'agents': ('ego_vehicle', 'other_vehicle', 'ego_vehicle_behind', 'other_vehicle_behind', 'infrastructure')
    }
    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 box_type_3d: str = 'LiDAR',
                 load_type: str = 'frame_based',
                 seq_length: int = 1,
                 present_idx: int = 1,
                 co_agents: Tuple[str] = ('ego_vehicle',),
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 with_velocity: bool = True,
                 adeptive_seq_length: bool = True,
                 scene_pipline: Optional[List[Union[dict, Callable]]] = [],
                 **kwargs) -> None:
        self.with_velocity = with_velocity
        self.adeptive_seq_length = adeptive_seq_length
        self.scene_pipline = Compose(scene_pipline)

        # TODO: Redesign multi-view data process in the future
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based')
        self.load_type = load_type

        assert seq_length > 0
        self.seq_length = seq_length
        assert present_idx in list(range(1, seq_length+1))
        self.present_idx = present_idx

        assert len(co_agents) > 0
        assert all(agent in self.METAINFO['agents'] for agent in co_agents) # type: ignore
        self.co_agents = co_agents

        assert box_type_3d.lower() in ('lidar', 'camera')
        super().__init__(
            data_root='',
            ann_file=ann_file,
            modality=modality,
            pipeline=pipeline,
            box_type_3d=box_type_3d, # type: ignore
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

    def get_ann_info(self, index: int) -> dict:
        data_info = self.get_data_info(index)
        co_agents = data_info['co_agents']
        seq_timestamps = data_info['seq_timestamps']
        ann_info = {timestamp : { agent : {} for agent in co_agents} for timestamp in seq_timestamps}
        for i, timestamp in enumerate(seq_timestamps):
            for j, agent in enumerate(co_agents):
                # test model
                input_dict = data_info['seq'][i][j]
                if 'ann_info' not in input_dict:
                    temp_info = self.parse_ann_info(input_dict)
                else:
                    temp_info = input_dict['ann_info']
                ann_info[timestamp][agent] = temp_info
        return ann_info

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        if self.load_type == 'mv_image_based':
            data_list = []
            for idx, (cam_id, img_info) in enumerate(info['images'].items()):
                camera_info = dict()
                camera_info['images'] = dict()
                camera_info['images'][cam_id] = img_info
                if 'cam_instances' in info and cam_id in info['cam_instances']:
                    camera_info['instances'] = info['cam_instances'][cam_id]
                else:
                    camera_info['instances'] = []
                # TODO: check whether to change sample_idx for 6 cameras
                #  in one frame
                camera_info['sample_idx'] = info['sample_idx'] * 6 + idx
                camera_info['ego2global'] = info['ego2global']

                if not self.test_mode:
                    # used in traing
                    camera_info['ann_info'] = self.parse_ann_info(camera_info)
                if self.test_mode and self.load_eval_anns:
                    camera_info['eval_ann_info'] = self.parse_ann_info(camera_info)
                data_list.append(camera_info)
            return data_list
        else:
            if self.modality['use_lidar']:
                info['num_pts_feats'] = info['lidar_points']['num_pts_feats']
                info['lidar_path'] = info['lidar_points']['lidar_path']
                info['lidar2ego'] = info['lidar_points']['lidar2ego']
            if not self.test_mode:
                # used in training
                info['ann_info'] = self.parse_ann_info(info)
            if self.test_mode and self.load_eval_anns:
                info['eval_ann_info'] = self.parse_ann_info(info)
            return info
        
    def parse_ann_info(self, info: dict) -> dict:

        ann_info = super().parse_ann_info(info)
        if ann_info is not None:
            if self.with_velocity:
                gt_bboxes_3d = ann_info['gt_bboxes_3d']
                gt_velocities = ann_info['velocities']
                nan_mask = np.isnan(gt_velocities[:, 0])
                gt_velocities[nan_mask] = [0.0, 0.0]
                gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocities], axis=-1) # 7 + 2
                ann_info['gt_bboxes_3d'] = gt_bboxes_3d
            else:
                gt_bboxes_3d = ann_info['gt_bboxes_3d']
                gt_bboxes_3d = np.concatenate([gt_bboxes_3d], axis=-1) # 7
                ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        else:
            # empty instance
            ann_info = dict()
            if self.with_velocity:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 9), dtype=np.float32)
            else:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

            if self.load_type in ['fov_image_based', 'mv_image_based']:
                ann_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                ann_info['gt_bboxes_labels'] = np.array(0, dtype=np.int64)
                ann_info['attr_labels'] = np.array(0, dtype=np.int64)
                ann_info['centers_2d'] = np.zeros((0, 2), dtype=np.float32)
                ann_info['depths'] = np.zeros((0), dtype=np.float32)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # TODO: Unify the coordinates
        if self.load_type in ['fov_image_based', 'mv_image_based']:
            gt_bboxes_3d = CameraInstance3DBoxes(
                ann_info['gt_bboxes_3d'],
                box_dim=ann_info['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5))
        else:
            gt_bboxes_3d = LiDARInstance3DBoxes(
                ann_info['gt_bboxes_3d'],
                box_dim=ann_info['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        ann_info['gt_bboxes_3d'] = gt_bboxes_3d

        return ann_info
    
    def load_data_list(self) -> List[dict]:
        annotations = mmengine.load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                             'keys')
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        total_scene_name = tuple(set([item['scene_name'] for item in raw_data_list]))

        scene_data = {scene_name : {agent : [] for agent in self.co_agents} for scene_name in total_scene_name}
        for raw_data in raw_data_list:
            vehicle_name = raw_data['vehicle_name']
            scene_name = raw_data['scene_name']
            if vehicle_name in self.co_agents:
                scene_data[scene_name][vehicle_name].append(self.parse_data_info(raw_data))

        data_list = []
        for scene_name, agent_data_dict in scene_data.items():
            data = {}
            data['scene_name'] = scene_name
            data['seq_length'] = self.seq_length
            data['present_idx'] = self.present_idx
            data['co_agents'] = self.co_agents
            data['co_length'] = len(self.co_agents)
            for agent in self.co_agents:
                agent_data_dict[agent] = sorted(agent_data_dict[agent], key=lambda x: x['timestamp'])
                if 'scene_timestamps' not in data.keys():
                    data['scene_timestamps'] = [agent_data_dict[agent][i]['timestamp'] for i in range(len(agent_data_dict[agent]))]
            data['scene_length'] = len(data['scene_timestamps'])
            if self.adeptive_seq_length and data['scene_length'] - self.seq_length + 1 <= 0:
                data['seq_length'] = data['scene_length']
            else:
                assert data['scene_length'] - self.seq_length + 1 > 0, f"The obtained sequence length is too long, maximum length {data['scene_length']}, required length {self.seq_length}"
            for i in range(data['seq_length'] - 1, data['scene_length']):
                frame = []
                seq_timestamps = []
                for j in range(i - data['seq_length'] + 1, i + 1):
                    frame.append([agent_data_dict[agent][j] for agent in self.co_agents])
                    seq_timestamps.append(data['scene_timestamps'][j])
                data_list.append({
                    'seq':frame,
                    'seq_timestamps':seq_timestamps
                })
                data_list[-1].update(data)
        return data_list
    
    def prepare_data(self, index: int) -> Union[dict, None]:
        data_info = self.get_data_info(index)
        co_agents = data_info['co_agents']
        seq_timestamps = data_info['seq_timestamps']
        seq = [ [ {} for j in range(len(co_agents))] for i in range(len(seq_timestamps))]
        for i in range(len(seq_timestamps)):
            for j in range(len(co_agents)):
                ori_input_dict = data_info['seq'][i][j]

                # deepcopy here to avoid inplace modification in pipeline.
                input_dict = copy.deepcopy(ori_input_dict)

                # box_type_3d (str): 3D box type.
                input_dict['box_type_3d'] = self.box_type_3d
                # box_mode_3d (str): 3D box mode.
                input_dict['box_mode_3d'] = self.box_mode_3d

                # pre-pipline return None to random another in `__getitem__`
                if not self.test_mode and self.filter_empty_gt:
                    if len(input_dict['ann_info']['gt_labels_3d']) == 0:
                        return None

                example: dict = self.pipeline(input_dict) # type: ignore

                if not self.test_mode and self.filter_empty_gt:
                    # after pipeline drop the example with empty annotations
                    # return None to random another in `__getitem__`
                    if example is None or len(
                            example['data_samples'].gt_instances_3d.labels_3d) == 0:
                        return None

                if self.show_ins_var:
                    if 'ann_info' in ori_input_dict:
                        self._show_ins_var(
                            ori_input_dict['ann_info']['gt_labels_3d'],
                            example['data_samples'].gt_instances_3d.labels_3d) # type: ignore
                    else:
                        print_log(
                            "'ann_info' is not in the input dict. It's probably that "
                            'the data is not in training mode',
                            'current',
                            level=WARNING)
                seq[i][j] = example
        data_info['example_seq'] = seq
        return self.scene_pipline(data_info)

    def get_cat_ids(self, idx: int) -> dict:
        data_info = self.get_data_info(idx)
        co_agents = data_info['co_agents']
        seq_timestamps = data_info['seq_timestamps']
        label_info = {timestamp : { agent : set() for agent in co_agents} for timestamp in seq_timestamps}
        for i, timestamp in enumerate(seq_timestamps):
            for j, agent in enumerate(co_agents):
                # test model
                input_dict = data_info['seq'][i][j]
                gt_labels = input_dict['ann_info']['gt_labels_3d'].tolist()
                label_info[timestamp][agent] = set(gt_labels)
        return label_info