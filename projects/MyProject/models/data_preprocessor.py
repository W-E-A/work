from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmdet3d.registry import MODELS
from mmdet3d.models import Det3DDataPreprocessor


@MODELS.register_module()
class DeepAccidentDataPreprocessor(Det3DDataPreprocessor):
    def __init__(self, *args, delete_pointcloud: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.delete_pointcloud = delete_pointcloud

    def forward(self,
                data: Union[dict, List[dict]],
                training: bool = False) -> dict:
        casted_scene_info = self.cast_data(data['scene_info']) # type: ignore
        seq_length = data['scene_info'][0].seq_length # type: ignore
        co_length = data['scene_info'][0].co_length # type: ignore
        present_idx = data['scene_info'][0].present_idx # type: ignore
        casted_example_seq = [ [ {} for j in range(co_length)] for i in range(seq_length)]
        for i in range(seq_length): # type: ignore
            for j in range(co_length):
                input_dict = data['example_seq'][i][j] # type: ignore
                casted_example_seq[i][j] = self.simple_process(input_dict, training) # type: ignore
                if self.delete_pointcloud:
                    if 'points' in casted_example_seq[i][j]['inputs'].keys():
                        casted_example_seq[i][j]['inputs'].pop('points')
                if i == present_idx:
                    if 'motion_label' in input_dict.keys(): # if motion
                        casted_example_seq[i][j]['motion_label'] = self.cast_data(input_dict['motion_label'])
                    if 'ego_motion_label' in input_dict.keys(): # if ego
                        casted_example_seq[i][j]['ego_motion_label'] = self.cast_data(input_dict['ego_motion_label'])
                    if 'inf_motion_label' in input_dict.keys(): # if inf
                        casted_example_seq[i][j]['inf_motion_label'] = self.cast_data(input_dict['inf_motion_label'])
                    if 'corr_heatmaps' in input_dict.keys(): # if corr
                        casted_example_seq[i][j]['corr_heatmaps'] = self.cast_data(input_dict['corr_heatmaps'])
                        assert 'corr_gt_masks' in input_dict.keys()
                        casted_example_seq[i][j]['corr_gt_masks'] = self.cast_data(input_dict['corr_gt_masks'])
                        assert 'corr_dilate_heatmaps' in input_dict.keys()
                        casted_example_seq[i][j]['corr_dilate_heatmaps'] = self.cast_data(input_dict['corr_dilate_heatmaps'])
                        assert 'corr_pos_nums' in input_dict.keys()
                        casted_example_seq[i][j]['corr_pos_nums'] = self.cast_data(input_dict['corr_pos_nums'])
        return {'scene_info': casted_scene_info, 'example_seq': casted_example_seq}