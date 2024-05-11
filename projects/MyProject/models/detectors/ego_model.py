from typing import Optional, Union, Dict, Sequence, List
from mmdet3d.registry import MODELS
from mmdet3d.models import MVXTwoStageDetector
from mmengine.device import get_device
import torch
from torch import Tensor
from mmengine.logging import print_log
import logging
def log(msg = "" ,level: int = logging.INFO):
    print_log(msg, "current", level)
from mmdet3d.structures import Det3DDataSample
from ...visualization import SimpleLocalVisualizer
from ...utils import warp_features
import copy
import numpy as np
import random
from ...utils.freeze_module import freeze_module

@MODELS.register_module()
class EgoModel(MVXTwoStageDetector):
    def __init__(self,
                 corr_model: Optional[dict] = None,
                 freeze_inf_model: bool=None,
                 pts_voxel_encoder: Optional[dict] = None,
                 pts_middle_encoder: Optional[dict] = None,
                 pts_fusion_layer: Optional[dict] = None,
                 pts_backbone: Optional[dict] = None,
                 pts_neck: Optional[dict] = None,
                #  temporal_backbone: Optional[dict] = None,
                 multi_task_head: Optional[dict] = None,
                #  train_comm_expand_layer: Optional[dict] = None,
                #  test_comm_expand_layer: Optional[dict] = None,
                 pts_train_cfg: Optional[dict] = None,
                 pts_test_cfg: Optional[dict] = None,
                 pts_fusion_cfg: Optional[dict] = None,
                 co_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None):
        super(EgoModel, self).__init__(
                pts_voxel_encoder=pts_voxel_encoder,
                pts_middle_encoder=pts_middle_encoder,
                pts_fusion_layer=pts_fusion_layer,
                pts_backbone=pts_backbone,
                pts_neck=pts_neck,
                data_preprocessor=data_preprocessor,
                init_cfg=init_cfg)
        if corr_model:
            self.corr_model = MODELS.build(corr_model)
        
        self.pts_train_cfg = pts_train_cfg
        self.pts_test_cfg = pts_test_cfg
        self.pts_fusion_cfg = pts_fusion_cfg
        self.co_cfg = co_cfg
        
        if multi_task_head:
            multi_task_head.update(train_cfg = pts_train_cfg)
            multi_task_head.update(test_cfg = pts_test_cfg)
            self.multi_task_head = MODELS.build(multi_task_head)
        if freeze_inf_model:
            for shared_module_name in self.pts_train_cfg.get('shared_weights',[]):
                items = shared_module_name.split('.')
                shared_module = self
                for item in items:
                    shared_module = getattr(shared_module, item)
                freeze_module(shared_module)
                print(f'Freeze: {shared_module_name} !!!!!!!!')

        if self.pts_train_cfg:
            self.train_mode = self.pts_train_cfg.get('train_mode', 'single') # type: ignore
            assert self.train_mode in ('single', 'fusion')

        if self.pts_test_cfg:
            pass

        if self.pts_fusion_cfg:
            self.corr_thresh = self.pts_fusion_cfg.get("corr_thresh", 0.2)
            self.train_ego_name = self.pts_fusion_cfg.get("train_ego_name", "ego_vehicle")
            self.test_ego_name = self.pts_fusion_cfg.get("test_ego_name", "ego_vehicle")
            self.pc_range = self.pts_fusion_cfg.get("pc_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
            self.warp_size = (self.pc_range[3], self.pc_range[4])

        if self.co_cfg:
            self.infrastructure_name = self.co_cfg.get('infrastructure_name', 'infrastructure')

    def extract_pts_feat(
        self,
        voxel_dict: Dict[str, Tensor],
        points: Optional[List[Tensor]] = None,
        batch_size: Optional[int] = None,
        img_feats: Optional[Sequence[Tensor]] = None,
        batch_input_metas: Optional[List[dict]] = None,
        extract_level: int = 4,
        return_voxel_features: bool = False,
        return_middle_features: bool = False,
        return_backbone_features: bool = False,
        return_neck_features:bool = True
        ) -> Dict[str, Union[Tensor, List[Tensor]]]:
        assert extract_level in list(range(1, 5))
        return_dict = {}
        voxel_features = self.pts_voxel_encoder(voxel_dict['voxels'], # [n*bs, N, feat]
                                                voxel_dict['num_points'], # [n*bs, ] realpoints
                                                voxel_dict['coors'], # [n*bs, 1 + 3] batch z y x
                                                img_feats,
                                                batch_input_metas) # FIXME 400MiB
        if return_voxel_features:
            return_dict['voxel_features'] = voxel_features
        if extract_level == 1:
            return return_dict

        if voxel_dict['coors'].shape[-1] < 4:
            assert batch_size != None
        else:
            temp_batch_size = voxel_dict['coors'][-1, 0] + 1 # type: ignore
            if batch_size != None:
                assert batch_size == temp_batch_size
            batch_size = temp_batch_size # type: ignore
        # [n*4, feat] # 65960 20 4
        middle_features = self.pts_middle_encoder(voxel_features, # type: ignore
                                    voxel_dict['coors'],
                                    batch_size) # return raw bev feature # FIXME 500MiB
        if return_middle_features:
            return_dict['middle_features'] = middle_features
        if extract_level == 2:
            return return_dict
        
        # [bs, feat, 1024, 1024] [N, C, H, W]
        backbone_features = self.pts_backbone(middle_features) # return tensor or sequence # type: ignore # FIXME backbone and neck 5000MiB
        if return_backbone_features:
            return_dict['backbone_features'] = backbone_features
        if extract_level == 3:
            return return_dict
        
        # list([bs, 128*3, 256, 256])
        neck_features = self.pts_neck(backbone_features) # Neck always return sequence # type: ignore # FIXME backbone and neck 5000MiB
        if return_neck_features:
            return_dict['neck_features'] = neck_features
        if extract_level == 4:
            return return_dict
        
        return return_dict
    

    def extract_feat(self,
                     batch_inputs_dict: dict,
                     batch_input_metas: List[dict],
                     **kwargs) -> Union[tuple, dict]:
        voxel_dict = batch_inputs_dict.get('voxels', None)
        points = batch_inputs_dict.get('points', None)
        pts_feat_dict = self.extract_pts_feat(
            voxel_dict,
            points=points,
            img_feats=None,
            batch_input_metas=batch_input_metas,
            **kwargs)
        return pts_feat_dict
    
    def forward(self,
            scene_info: Sequence,
            example_seq: Sequence,
            mode: str = 'tensor',
            **kwargs) -> Union[Dict[str, torch.Tensor], list]:
        
        assert mode in ('loss', 'predict')
        scene_info_0 = scene_info[0]
        batch_size = len(scene_info)
        seq_length = scene_info_0.seq_length
        present_idx = scene_info_0.present_idx
        co_agents = scene_info_0.co_agents
        co_length = scene_info_0.co_length
        sample_idx = scene_info_0.sample_idx
        scene_name = scene_info_0.scene_name
        seq_timestamps_0 = scene_info_0.seq_timestamps[0]
        save_dir = f'{scene_name}_{seq_timestamps_0}'
        self.infrastructure_id = co_agents.index(self.infrastructure_name)
        self.ego_id = co_agents.index(self.train_ego_name)
        present_seq = example_seq[present_idx]

        if self.train_mode == 'single':
            # ego的所有输入
            input_dict_ego = present_seq[self.ego_id]['inputs'] # voxel batch
            input_samples_ego = present_seq[self.ego_id]['data_samples'] # batch
            ego_metas = [sample.metainfo for sample in input_samples_ego] # batch
            pts_feat_dict_ego = self.extract_feat(
                input_dict_ego,
                ego_metas,
                extract_level = 4,
                return_voxel_features = False,
                return_middle_features = False,
                return_backbone_features = False,
                return_neck_features = True) # FIXME 4 times
            ego_features = pts_feat_dict_ego['neck_features'] # B C H W
            # gt
            ego_instances = []
            for samples in input_samples_ego:
                valid_mask = samples.gt_instances_3d.bbox_3d_isvalid
                ego_instances.append(samples.gt_instances_3d[valid_mask]) # visible targets only
        else:
            # ego的所有输入
            input_dict_ego = present_seq[self.ego_id]['inputs'] # voxel batch
            input_samples_ego = present_seq[self.ego_id]['data_samples'] # batch
            ego_metas = [sample.metainfo for sample in input_samples_ego] # batch
            pts_feat_dict_ego = self.extract_feat(
                input_dict_ego,
                ego_metas,
                extract_level = 4,
                return_voxel_features = False,
                return_middle_features = False,
                return_backbone_features = False,
                return_neck_features = True) # FIXME 4 times
            ego_features = pts_feat_dict_ego['neck_features'] # B C H W

            # infrastructure的所有输入
            input_dict_inf = present_seq[self.infrastructure_id]['inputs'] # voxel batch
            input_samples_inf = present_seq[self.infrastructure_id]['data_samples'] # batch
            infrastructure_metas = [sample.metainfo for sample in input_samples_inf] # batch
            pts_feat_dict_inf = self.corr_model.extract_feat(
                input_dict_inf,
                infrastructure_metas,
                extract_level = 4,
                return_voxel_features = False,
                return_middle_features = False,
                return_backbone_features = False,
                return_neck_features = True) # FIXME 4 times
            infrastructure_features = pts_feat_dict_inf['neck_features'] # B C H W

            # gt
            ego_coop_instances = [samples.gt_instances_3d for samples in input_samples_ego] # 1*B
            for b in range(batch_size):
                ego_coop_instances[b].coop_isvalid = ego_coop_instances[b].bbox_3d_isvalid
            inf_coop_instances = [samples.gt_instances_3d for samples in input_samples_inf] # 1*B
            for b in range(batch_size):
                ego_track_id = ego_coop_instances[b].track_id
                # other visible 
                other_track_id = inf_coop_instances[b].track_id
                in_mask = np.isin(ego_track_id, other_track_id)
                new_isvalid = copy.deepcopy(ego_coop_instances[b].coop_isvalid)
                new_isvalid[in_mask] = True
                ego_coop_instances[b].coop_isvalid = new_isvalid # global valid bboxes for ego
            coop_instances = []
            for instance in ego_coop_instances: # type: ignore
                coop_instances.append(instance[instance.coop_isvalid])

        if mode == 'loss':
            if self.train_mode == 'single':
                det_forward_kwargs = {}
                ego_feat_dict = self.multi_task_head(ego_features,det_forward_kwargs=det_forward_kwargs) 
                heatmaps, anno_boxes, inds, masks = self.multi_task_head.det_head.get_targets(ego_instances)
                det_loss_kwargs = {
                'heatmaps':heatmaps,# necessary
                'anno_boxes':anno_boxes,# necessary
                'inds':inds,# necessary
                'masks':masks,# necessary
                }
                loss_dict = self.multi_task_head.loss(ego_feat_dict,
                                                    det_loss_kwargs=det_loss_kwargs)
                return loss_dict
            else: #fusion
                #prepare motion label
                infrastructure_label = present_seq[self.infrastructure_id]['inf_motion_label'] # motion_label also for ego single
                infrastructure_label, infrastructure_input = self.corr_model.multi_task_head.motion_head.prepare_future_labels(infrastructure_label)
                ego_motion_labels = [present_seq[self.ego_id]['ego_motion_label']]
                ego_motion_labels, ego_motion_inputs = self.corr_model.multi_task_head.corr_head.prepare_ego_labels(ego_motion_labels)
                #获得路端推理结果
                det_forward_kwargs = {}
                motion_forward_kwargs = {
                    'future_distribution_inputs':infrastructure_input,
                    'noise':None
                }
                corr_forward_kwargs = {
                    'ego_motion_inputs':ego_motion_inputs
                }
                infrastructure_feat_dict = self.corr_model.multi_task_head(
                    infrastructure_features,
                    det_forward_kwargs=det_forward_kwargs,
                    motion_forward_kwargs=motion_forward_kwargs,
                    corr_forward_kwargs=corr_forward_kwargs,
                )
                #得到相关性heatmap 以此筛选出协调区域
                gt_corr_heatmaps = present_seq[self.infrastructure_id]['corr_heatmaps']
                for idx in range(len(gt_corr_heatmaps)):
                    gt_corr_heatmaps[idx] = gt_corr_heatmaps[idx][self.ego_id,:,:]
                gt_corr_heatmaps = torch.stack(gt_corr_heatmaps, dim=0).unsqueeze(1)
                pred_corr_heatmap = infrastructure_feat_dict['corr_feat'][0][0]['heatmap'].sigmoid()
                corr_mask = gt_corr_heatmaps > self.corr_thresh

                #对路端特帧进行位姿变换
                present_pose_matrix = []
                for b in range(batch_size):
                    present_pose_matrix.append(scene_info[b].pose_matrix[present_idx, self.infrastructure_id, self.ego_id, ...]) # use ego to other1, other2, ... # type: ignore
                present_pose_matrix = torch.tensor(present_pose_matrix)
                infrastructure_feature = infrastructure_features[0] # C, H, W
                infrastructure_feature = corr_mask.float() * infrastructure_feature
                warp_infra_feat = warp_features(infrastructure_feature, present_pose_matrix, self.warp_size) #B C H W
                warp_corr_mask = warp_features(corr_mask.float(), present_pose_matrix, self.warp_size).bool() #B C H W
                #融合
                ego_fusion_result = self.pts_fusion_layer(ego_features[0], warp_infra_feat, warp_corr_mask) # B C H W

                #fusion det loss
                det_forward_kwargs = {}
                fusion_feat_dict = self.multi_task_head(ego_fusion_result,det_forward_kwargs=det_forward_kwargs) 
                heatmaps, anno_boxes, inds, masks = self.multi_task_head.det_head.get_targets(coop_instances)
                det_loss_kwargs = {
                'heatmaps':heatmaps,# necessary
                'anno_boxes':anno_boxes,# necessary
                'inds':inds,# necessary
                'masks':masks,# necessary
                }
                loss_dict = self.multi_task_head.loss(fusion_feat_dict,
                                                    det_loss_kwargs=det_loss_kwargs)
                return loss_dict
        else:
            if self.train_mode == 'single':
                det_forward_kwargs = {}
                ego_feat_dict = self.multi_task_head(ego_features,det_forward_kwargs=det_forward_kwargs)
                det_pred_kwargs = {
                'batch_input_metas':ego_metas
                }
                predict_dict = self.multi_task_head.predict(
                ego_feat_dict,
                det_pred_kwargs=det_pred_kwargs)
                if 'det_pred' in predict_dict:
                    det_ret_list = []
                    pred_result = predict_dict['det_pred'] # add to pred_instances_3d from None to instance of bboxes_3d scores_3d labels_3d
                    for b in range(batch_size):
                        sample = Det3DDataSample()
                        sample.set_metainfo(
                            dict(
                                scene_sample_idx = scene_info[b].sample_idx,
                                scene_name = scene_info[b].scene_name,
                                agent_name = self.test_ego_name, # FIXME
                                sample_idx = ego_metas[b]['sample_idx'], # type: ignore
                                box_type_3d = ego_metas[b]['box_type_3d'], # type: ignore
                                lidar_path = ego_metas[b]['lidar_path'], # type: ignore
                            )
                        )
                        sample.gt_instances_3d = ego_instances[b] # type: ignore
                        sample.gt_instances_3d.pop('track_id') # no need array
                        sample.gt_instances_3d.pop('bbox_3d_isvalid') # no need array
                        # sample.gt_instances_3d.pop('coop_isvalid') # no need array
                        # sample.gt_instances_3d.pop('correlations') # no need array
                        sample.pred_instances_3d = pred_result[b]
                        det_ret_list.append(sample)
                return det_ret_list
            else:
                return []
            
