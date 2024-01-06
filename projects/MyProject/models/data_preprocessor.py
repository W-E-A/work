from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmdet3d.registry import MODELS
from mmdet3d.models import Det3DDataPreprocessor

@MODELS.register_module()
class DeepAccidentDataPreprocessor(Det3DDataPreprocessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,
                data: Union[dict, List[dict]],
                training: bool = False) -> dict:
        casted_scene_info = self.cast_data(data['scene_info']) # type: ignore
        seq_length = data['scene_info'][0].seq_length # type: ignore
        co_length = data['scene_info'][0].co_length # TODO # type: ignore
        casted_example_seq = [ [ {} for j in range(co_length)] for i in range(seq_length)]
        for i in range(seq_length): # type: ignore
            for j in range(co_length):
                input_dict = data['example_seq'][i][j] # type: ignore
                casted_example_seq[i][j] = self.points_process(input_dict, training) # type: ignore
        return {'scene_info': casted_scene_info, 'example_seq': casted_example_seq}
    
    def points_process(self, data: dict, training: bool = False) -> dict:
        data = self.cast_data(data) # type: ignore
        inputs, single_samples, coop_samples = data['inputs'], data['single_samples'], data['coop_samples']
        batch_inputs = dict()
        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']

            if self.voxel:
                voxel_dict = self.voxelize(inputs['points'], coop_samples)
                batch_inputs['voxels'] = voxel_dict

        return {'inputs': batch_inputs, 'single_samples': single_samples, 'coop_samples': coop_samples,}