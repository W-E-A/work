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
        co_length = data['scene_info'][0].co_length # FIXME # type: ignore
        casted_example_seq = [ [ {} for j in range(co_length)] for i in range(seq_length)]
        for i in range(seq_length): # type: ignore
            for j in range(co_length):
                input_dict = data['example_seq'][i][j] # type: ignore
                casted_example_seq[i][j] = self.simple_process(input_dict, training) # type: ignore
        return {'scene_info': casted_scene_info, 'example_seq': casted_example_seq}