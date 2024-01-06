from typing import List, Union, Optional, Sequence, Tuple
from mmdet.models.utils import multi_apply
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor


# @MODELS.register_module()
# class MotionHead(BaseModule):
#     def __init__(self,
#                  in_channels: int = 128,
#                  inner_channels: Optional[int] = None,
#                  history_length: int = 3,
#                  future_length: int = 1,
#                  future_discount: float = 0.95,
#                  future_dim: int = 6,
#                  common_heads: Optional[List[dict]] = None,
#                  num_gru_blocks: int = 3,
#                  prob_enable: bool = True,
#                  prob_on_foreground: bool = False,
#                  prob_spatially: bool = False,
#                  prob_each_future: bool = False,
#                  prob_latent_dim: int = 32,
#                  distribution_log_sigmas: tuple = (-5.0, 5.0),
#                  detach_state=True,
#                  norm_cfg: dict = dict(type='BN2d'),
#                  init_bias: float = -2.19,
#                  train_cfg: Optional[dict] = None,
#                  test_cfg: Optional[dict] = None,
#                  init_cfg: Union[dict, List[dict], None] = None,

#                  class_weights=None,
#                  use_topk=True,
#                  topk_ratio=0.25,
#                  ignore_index=255,
#                  posterior_with_label=False,
#                  sample_ignore_mode='all_valid',
#                  using_focal_loss=False,
#                  focal_cfg=dict(type='mmdet.GaussianFocalLoss', reduction='none'),
#                  loss_weights=None,
#                  ):
#         super(MotionHead, self).__init__(init_cfg=init_cfg)
#         # FIXME

#         self.in_channels = in_channels
#         self.inner_channels = inner_channels
#         self.history_length = history_length
#         self.future_length = future_length
#         self.future_discount = future_discount
#         self.future_dim = future_dim
#         self.train_cfg = train_cfg
#         self.test_cfg = test_cfg
    
#     def forward_single(self, x: Tensor, ) -> List[dict]:
#         N, C, H, W = x.shape
#         if self.future_length > 0:
#             pass
#         else:
#             pass

#     def forward(self, feats: Union[Tensor, List[Tensor]], ) -> Tuple[List[Tensor]]:
#         if not isinstance(feats, Sequence):
#             feats = [feats]
#         return multi_apply(self.forward_single, feats, ) # type: ignore


@MODELS.register_module()
class MTHead(BaseModule):
    def __init__(
            self,
            det_head: Optional[dict] = None,
            motion_head:Optional[dict] = None,
            train_cfg: Optional[dict] = None,
            test_cfg: Optional[dict] = None,
            init_cfg: Union[dict, List[dict], None] = None,
            ):
        super().__init__(init_cfg=init_cfg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if det_head:
            det_head.update(train_cfg=self.train_cfg)
            det_head.update(test_cfg=self.test_cfg)
            self.det_head = MODELS.build(det_head)
        if motion_head:
            motion_head.update(train_cfg=self.train_cfg)
            motion_head.update(test_cfg=self.test_cfg)
            self.motion_head = MODELS.build(motion_head)

    def forward(self, feats: Union[Tensor, List[Tensor]]) -> dict:
        return_dict = {}
        if not isinstance(feats, Sequence):
            feats = [feats]
        if self.det_head:
            multi_tasks_multi_feats: Tuple[List[Tensor]] = self.det_head(feats)
            return_dict['det_feat'] = multi_tasks_multi_feats # [[t1,],[t2,],[...]]
        if self.motion_head:
            multi_tasks_multi_feats: Tuple[List[Tensor]] = self.motion_head(feats)
            return_dict['motion_feat'] = multi_tasks_multi_feats

        return return_dict

    def loss(self,
             feat_dict: dict,
             batch_det_instances: Optional[List[InstanceData]] = None,
             batch_motion_instances: Optional[List[InstanceData]] = None,
             gather_task_loss: bool = False
             ) -> dict:

        loss_dict = {}
        
        if 'det_feat' in feat_dict and batch_det_instances != None:
            multi_tasks_multi_feats = feat_dict['det_feat']
            # TODO just one scale here
            temp_dict = self.det_head.loss_by_feat(multi_tasks_multi_feats, batch_det_instances)
            if gather_task_loss:
                gather_dict = {}
                for key in temp_dict.keys():
                    if 'loss' in key:
                        cut_key = key.split('.')[-1]
                        if cut_key not in gather_dict:
                            gather_dict[cut_key] = []
                        gather_dict[cut_key].append(temp_dict[key])
                loss_dict.update(gather_dict)
            else:
                loss_dict.update(temp_dict)

        if 'motion_feat' in feat_dict and batch_motion_instances != None:
            multi_tasks_multi_feats = feat_dict['motion_feat']
            # TODO just one scale here
            temp_dict = self.motion_head.loss_by_feat(multi_tasks_multi_feats, batch_motion_instances)
            if gather_task_loss:
                gather_dict = {}
                for key in temp_dict.keys():
                    if 'loss' in key:
                        cut_key = key.split('.')[-1]
                        if cut_key not in gather_dict:
                            gather_dict[cut_key] = []
                        gather_dict[cut_key].append(temp_dict[key])
                loss_dict.update(gather_dict)
            else:
                loss_dict.update(temp_dict)
        
        return loss_dict

    def predict(self,
             feat_dict: dict,
             batch_input_metas: Optional[List[dict]] = None,
             ) -> dict:

        predict_dict = {}
        
        if 'det_feat' in feat_dict and batch_input_metas != None:
            multi_tasks_multi_feats = feat_dict['det_feat']
            # TODO just one scale here
            batch_result_instances = self.det_head.predict_by_feat(multi_tasks_multi_feats, batch_input_metas)
            predict_dict['det_pred'] = batch_result_instances

        # if 'motion_feat' in feat_dict and batch_input_metas != None:
        #     multi_tasks_multi_feats = feat_dict['motion_feat']
        #     # TODO just one scale here
        #     batch_result_instances = self.motion_head.predict_by_feat(multi_tasks_multi_feats, batch_input_metas)
        #     predict_dict['motion_pred'] = batch_result_instances

        return predict_dict
