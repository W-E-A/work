from typing import Callable, List, Optional, Union, Tuple
from mmdet3d.registry import DATASETS
from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmengine.dataset import Compose
import mmengine
from mmengine.logging import print_log
from logging import WARNING
import numpy as np
from mmdet3d.structures import LiDARInstance3DBoxes, CameraInstance3DBoxes
import copy

@DATASETS.register_module()
class DeepAccident_V2X_Dataset(Det3DDataset):
    """
    dict_keys(['seq', 'example_seq', 'scene_name', 'seq_length', 'present_idx',
    'co_agents', 'scene_length', 'scene_timestamps', 'sample_idx',
    'seq_timestamps', 'co_length',
    ])

    对于数据生成标明有场景来源，代理列表，序列长度，序列时间戳的数据
    
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
                 key_interval: int = 1,
                 seq_length: int = 1,
                 present_idx: int = 1,
                 co_agents: Tuple[str] = ('ego_vehicle',),
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 scene_shuffle: bool = False,
                 with_velocity: bool = True,
                 adeptive_seq_length: bool = True,
                 scene_pipline: Optional[List[Union[dict, Callable]]] = [],
                 **kwargs) -> None:
        self.scene_shuffle = scene_shuffle
        self.with_velocity = with_velocity
        self.adeptive_seq_length = adeptive_seq_length
        self.scene_pipline = Compose(scene_pipline)

        # TODO: Redesign multi-view data process in the future
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based')
        self.load_type = load_type

        assert key_interval > 0
        self.key_interval = key_interval
        assert seq_length > 0
        self.seq_length = seq_length
        assert present_idx in list(range(0, seq_length))
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

                if self.load_eval_anns or not self.test_mode:
                    camera_info['ann_info'] = self.parse_ann_info(camera_info)
                data_list.append(camera_info)
            return data_list
        else:
            if self.modality['use_lidar']:
                info['num_pts_feats'] = info['lidar_points']['num_pts_feats']
                info['lidar_path'] = info['lidar_points']['lidar_path']
                info['lidar2ego'] = info['lidar_points']['lidar2ego']
            if self.load_eval_anns or not self.test_mode:
                info['ann_info'] = self.parse_ann_info(info)
            return info
        
    def parse_ann_info(self, info: dict) -> dict:

        ann_info = super().parse_ann_info(info)
        if ann_info is not None:
            if self.with_velocity:
                gt_bboxes_3d = ann_info['gt_bboxes_3d']
                gt_velocities = ann_info['velocities']
                nan_mask = np.isnan(gt_velocities[:, 0])
                gt_velocities[nan_mask] = np.array([0.0, 0.0])
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

        total_scene_name = list(set([item['scene_name'] for item in raw_data_list]))
        if not self.scene_shuffle:
            def custom_sort(s):
                parts = s.split('_')
                return (parts[2], parts[3], parts[4])
            total_scene_name = sorted(total_scene_name, key=custom_sort)

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
            for i in range(data['seq_length'] - 1, data['scene_length'], self.key_interval): # 5 6 7 8 9 10 11 12 13 14
                frame = []
                seq_timestamps = []
                for j in range(i - data['seq_length'] + 1, i + 1): # 0~6 9~15
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
            for j, agent in enumerate(co_agents):
                ori_input_dict = data_info['seq'][i][j]

                # deepcopy here to avoid inplace modification in pipeline.
                input_dict = copy.deepcopy(ori_input_dict)

                input_dict['agent'] = agent

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
    