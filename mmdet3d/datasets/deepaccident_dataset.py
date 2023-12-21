from typing import Callable, List, Optional, Sequence, Union,  Dict, Tuple, Mapping, Any
from mmdet3d.registry import DATASETS
from mmengine.dataset import BaseDataset
from mmengine.config import Config
import mmengine

@DATASETS.register_module()
class DeepAccident_V2X_Dataset(BaseDataset):
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
        for scene_name, agent_data in scene_data.items():
            scene_data = {}

            temp_name = self._metainfo['agents'][0]
            timestamps = [frame['timestamp'] for frame in agent_data if frame['vehicle_name'] == temp_name]
            timestamps.sort()

            temp_data = agent_data[0]

            scene_data['name'] = scene_name
            scene_data['metainfo'] = temp_data['scenario_meta']
            scene_data['seq_timestamps'] = timestamps
            scene_data['seq_interval'] = temp_data['sample_interval']
            scene_data['seq_length'] = len(timestamps)
            scene_data['agents'] = self._metainfo['agents']
            scene_data['seq_data'] = {timestamp: {agent : {} for agent in self._metainfo['agents']} for timestamp in timestamps}

            for frame in agent_data:
                agent = frame['vehicle_name']
                timestamp = frame['timestamp']

                scene_data['seq_data'][timestamp][agent]['lidar_prefix'] = frame['lidar_prefix']
                scene_data['seq_data'][timestamp][agent]['lidar_path'] = frame['lidar_path']
                scene_data['seq_data'][timestamp][agent]['bev_path'] = frame['bev_path']
                scene_data['seq_data'][timestamp][agent]['cams'] = frame['cams']
                scene_data['seq_data'][timestamp][agent]['lidar_to_ego_matrix'] = frame['lidar_to_ego_matrix']
                scene_data['seq_data'][timestamp][agent]['ego_to_world_matrix'] = frame['ego_to_world_matrix']
                scene_data['seq_data'][timestamp][agent]['vehicle_speed_x'] = frame['vehicle_speed_x']
                scene_data['seq_data'][timestamp][agent]['vehicle_speed_y'] = frame['vehicle_speed_y']

                if 'gt_names' not in scene_data['seq_data'][timestamp].keys():
                    scene_data['seq_data'][timestamp]['gt_names'] = frame['gt_names']
                if 'gt_boxes' not in scene_data['seq_data'][timestamp].keys():
                    scene_data['seq_data'][timestamp]['gt_boxes'] = frame['gt_boxes']
                if 'gt_velocity' not in scene_data['seq_data'][timestamp].keys():
                    scene_data['seq_data'][timestamp]['gt_velocity'] = frame['gt_velocity']
                if 'vehicle_id' not in scene_data['seq_data'][timestamp].keys():
                    scene_data['seq_data'][timestamp]['vehicle_id'] = frame['vehicle_id']
                if 'num_lidar_pts' not in scene_data['seq_data'][timestamp].keys():
                    scene_data['seq_data'][timestamp]['num_lidar_pts'] = frame['num_lidar_pts']
                if 'camera_visibility' not in scene_data['seq_data'][timestamp].keys():
                    scene_data['seq_data'][timestamp]['camera_visibility'] = frame['camera_visibility']

            data_list.append(scene_data)
        return data_list