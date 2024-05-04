import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from ._base_motion_head import BaseMotionHead
from ..modules.motion_modules import ResFuturePrediction, ResFuturePredictionV2
from ...utils import predict_instance_segmentation_and_trajectories


@MODELS.register_module()
class IterativeFlow(BaseMotionHead):
    def __init__(
        self,
        detach_state=True,
        n_gru_blocks=1,
        using_v2=False,
        flow_warp=True,
        **kwargs,
    ):
        super(IterativeFlow, self).__init__(**kwargs)

        if using_v2:
            self.future_prediction = ResFuturePredictionV2(
                in_channels=self.in_channels,
                latent_dim=self.prob_latent_dim,
                n_future=self.n_future,
                detach_state=detach_state,
                n_gru_blocks=n_gru_blocks,
                flow_warp=flow_warp,
            )
        else:
            self.future_prediction = ResFuturePrediction(
                in_channels=self.in_channels,
                latent_dim=self.prob_latent_dim,
                n_future=self.n_future,
                detach_state=detach_state,
                n_gru_blocks=n_gru_blocks,
                flow_warp=flow_warp,
            )


    def forward(self, bevfeats, future_distribution_inputs=None, noise=None):
        '''
        the forward process of motion head:
        1. get present & future distributions
        2. iteratively get future states with ConvGRU
        3. decode present & future states with the decoder heads
        '''
        # import pdb;pdb.set_trace()
        bevfeats = bevfeats[0] # b, 384, 256, 256 输入应该是结合了历史bev信息也就是temporal模块的bev特征，或者是单帧的BEV特征
        bevfeats = self.cropper(bevfeats) # b, 384, 200, 200

        if self.training:
            assert future_distribution_inputs is not None
        
        # future_distribution_inputs  B len 1+1+2+2 200, 200 or None

        if not self.training:
            res = list()
            if self.n_future > 0:
                present_state = bevfeats.unsqueeze(dim=1).contiguous()
                
                # sampling probabilistic distribution
                samples, output_distribution = self.distribution_forward(
                    present_state, future_distribution_inputs, noise
                )

                b, _, _, h, w = present_state.shape
                hidden_state = present_state[:, 0]

                for sample in samples:
                    res_single = {}
                    future_states = self.future_prediction(sample, hidden_state)
                    future_states = torch.cat([present_state, future_states], dim=1)
                    # flatten dimensions of (batch, sequence)
                    batch, seq = future_states.shape[:2]
                    flatten_states = future_states.flatten(0, 1)

                    if self.training:
                        res_single.update(output_distribution)

                    for task_key, task_head in self.task_heads.items():
                        res_single[task_key] = task_head(
                            flatten_states).view(batch, seq, -1, h, w)

                    res.append(res_single)
            else:
                b, _, h, w = bevfeats.shape
                for task_key, task_head in self.task_heads.items():
                    res[task_key] = task_head(bevfeats).view(b, 1, -1, h, w)
        else:
            res = {}
            if self.n_future > 0:
                present_state = bevfeats.unsqueeze(dim=1).contiguous() # b, 1, 384, 200, 200
                # import pdb;pdb.set_trace()
                # sampling probabilistic distribution
                sample, output_distribution = self.distribution_forward(
                    present_state, future_distribution_inputs, noise # B len 1+1+2+2 200, 200
                )

                b, _, _, h, w = present_state.shape
                hidden_state = present_state[:, 0] # b 384 200 200

                future_states = self.future_prediction(sample, hidden_state) # GRUS and so on, input B, 1, dim, 200, 200 and B, 384, 200, 200 output 1,5,384,200,200
                future_states = torch.cat([present_state, future_states], dim=1) #(B,1,384,200,200) cat (B,5,384,200,200)
                # flatten dimensions of (batch, sequence)
                batch, seq = future_states.shape[:2]
                flatten_states = future_states.flatten(0, 1) #(B*6,384,200,200)

                if self.training:
                    res.update(output_distribution)

                for task_key, task_head in self.task_heads.items():
                    res[task_key] = task_head(
                        flatten_states).view(batch, seq, -1, h, w) # for seg? (B*6,384,200,200) -> (B*6,2,200,200) -> (B,6,2,200,200)
            else:
                b, _, h, w = bevfeats.shape
                for task_key, task_head in self.task_heads.items():
                    res[task_key] = task_head(bevfeats).view(b, 1, -1, h, w)
        return res


    def predict_by_feat(self, predictions: list):
        # ['segmentation', 'instance_flow', 'instance_center', 'instance_offset'] 2 2 1 2
        # output future seg and ins-seg, not traj
        ret_list = []
        for pred in predictions:
            seg_prediction = torch.argmax(
                pred['segmentation'], dim=2, keepdims=True) # B, len, 1, H, W

            if self.using_focal_loss:
                pred['instance_center'] = torch.sigmoid(
                    pred['instance_center']) # B, len, 1, H, W

            pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
                pred, compute_matched_centers=False,
            )
            ret_list.append((seg_prediction, pred_consistent_instance_seg))

        return ret_list