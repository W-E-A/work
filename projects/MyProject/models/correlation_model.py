from typing import Optional, Union, Dict, Sequence, List
from mmdet3d.registry import MODELS
from mmdet3d.models import MVXTwoStageDetector
from mmengine.device import get_device
import torch
from torch import Tensor
from mmengine.logging import print_log
import logging
from ..visualization import SimpleLocalVisualizer
from mmdet3d.structures import Det3DDataSample


def log(msg = "" ,level: int = logging.INFO):
    print_log(msg, "current", level)
@MODELS.register_module()
class CorrelationModel(MVXTwoStageDetector):
    def __init__(self,
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
                #  pts_fusion_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None):
        super(CorrelationModel, self).__init__(
                pts_voxel_encoder=pts_voxel_encoder,
                pts_middle_encoder=pts_middle_encoder,
                pts_fusion_layer=pts_fusion_layer,
                pts_backbone=pts_backbone,
                pts_neck=pts_neck,
                data_preprocessor=data_preprocessor,
                init_cfg=init_cfg)
        
        self.pts_train_cfg = pts_train_cfg
        self.pts_test_cfg = pts_test_cfg
        
        if multi_task_head:
            multi_task_head.update(train_cfg = pts_train_cfg)
            multi_task_head.update(test_cfg = pts_test_cfg)
            self.multi_task_head = MODELS.build(multi_task_head)
        
        # if self.pts_train_cfg:
        #     self.gather_task_loss = self.pts_train_cfg.get('gather_task_loss', True) # type: ignore
        #     self.train_mode = self.pts_train_cfg.get('train_mode', 'single') # type: ignore
        #     assert self.train_mode in ('single', 'sparse_fusion', 'dense_fusion')

        # if self.pts_test_cfg:
        #     self.test_mode = self.pts_test_cfg.get('test_mode', 'single') # type: ignore
        #     self.score_threshold = self.pts_test_cfg.get('score_threshold', 0.1)
        #     assert self.test_mode in ('full', 'where2comm', 'new_method', 'single')



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
                                                batch_input_metas)
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
                                    batch_size) # return raw bev feature
        if return_middle_features:
            return_dict['middle_features'] = middle_features
        if extract_level == 2:
            return return_dict
        
        # [bs, feat, 1024, 1024] [N, C, H, W]
        backbone_features = self.pts_backbone(middle_features) # return tensor or sequence # type: ignore
        if return_backbone_features:
            return_dict['backbone_features'] = backbone_features
        if extract_level == 3:
            return return_dict
        
        # list([bs, 128*3, 256, 256])
        neck_features = self.pts_neck(backbone_features) # Neck always return sequence # type: ignore
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
        # imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        # img_feats = self.extract_img_feat(imgs, batch_input_metas)
        # pts_feats = self.extract_pts_feat(
        #     voxel_dict,
        #     points=points,
        #     img_feats=img_feats,
        #     batch_input_metas=batch_input_metas)
        # return (img_feats, pts_feats)
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
        self.ego_id = co_agents.index('ego_vehicle') # FIXME
        self.inf_id = co_agents.index('infrastructure') # FIXME
        present_seq = example_seq[present_idx]

        ################################ INPUT DEBUG (stop here)################################
        # scene_info_0.pop('pose_matrix')
        # scene_info_0.pop('future_motion_matrix')
        # scene_info_0.pop('loc_matrix')
        # log(scene_info_0)
        # visualizer: SimpleLocalVisualizer = SimpleLocalVisualizer.get_current_instance()
        # assert batch_size == 1
        # visualizer.set_points(present_seq[self.inf_id]['inputs']['points'][0].cpu().numpy())
        # visualizer.just_save('./data/temp.png')
        # import pdb
        # pdb.set_trace()
        # if mode == 'loss': 
        #     return {'fakeloss' : torch.ones(1, dtype=torch.float32, device=get_device(), requires_grad=True)}
        # else:
        #     return []
        ################################ INPUT DEBUG (stop here)################################

        neck_features = []
        ins_list = []
        meat_list = []
        for j, agent in enumerate(co_agents):
            # batch个单车检测的输入点云和对应的gt
            input_dict = present_seq[j]['inputs'] # voxel batch
            input_samples = present_seq[j]['data_samples'] # batch
            input_metas = [sample.metainfo for sample in input_samples] # batch

            pts_feat_dict = self.extract_feat(
                input_dict,
                input_metas,
                extract_level = 4,
                return_voxel_features = False,
                return_middle_features = False,
                return_backbone_features = False,
                return_neck_features = True)
            
            pts_neck_features = pts_feat_dict['neck_features'] # B C H W
            neck_features.append(pts_neck_features)

            batch_visible_instances = []
            for samples in input_samples:
                valid_mask = samples.gt_instances_3d.bbox_3d_isvalid
                batch_visible_instances.append(samples.gt_instances_3d[valid_mask]) # A*B visible
            ins_list.append(batch_visible_instances)
            meat_list.append(input_metas)
        
        if mode == 'loss':
            single_head_feat_dict = self.multi_task_head(neck_features[self.inf_id]) # out from dethead and motionhead

            heatmaps, anno_boxes, inds, masks = self.multi_task_head.det_head.get_targets(ins_list[self.inf_id]) # B
            single_det_gt = {
                    'heatmaps':heatmaps,
                    'anno_boxes':anno_boxes,
                    'inds':inds,
                    'masks':masks,
                }
            single_det_loss_dict = self.multi_task_head.loss(single_head_feat_dict,
                                                det_gt = single_det_gt,
                                                motion_gt = None,
                                                gather_task_loss = False) # FIXME
            return single_det_loss_dict
        else:
            single_head_feat_dict = self.multi_task_head(neck_features[self.inf_id]) # out from dethead and motionhead # FIXME only forward ego feature
            
            ret_list = []
            predict_dict = self.multi_task_head.predict(single_head_feat_dict,  meat_list[self.inf_id])
            if 'det_pred' in predict_dict:
                pred_result = predict_dict['det_pred'] # add to pred_instances_3d from None to instance of bboxes_3d scores_3d labels_3d
                for b in range(batch_size):
                    sample = Det3DDataSample()
                    sample.set_metainfo(
                        dict(
                            scene_sample_idx = scene_info[b].sample_idx,
                            scene_name = scene_info[b].scene_name,
                            agent_name = 'infrastructure', # FIXME
                            sample_idx = meat_list[self.inf_id][b]['sample_idx'], # type: ignore
                            box_type_3d = meat_list[self.inf_id][b]['box_type_3d'], # type: ignore
                            lidar_path = meat_list[self.inf_id][b]['lidar_path'], # type: ignore
                        )
                    )
                    sample.gt_instances_3d = ins_list[self.inf_id][b] # type: ignore
                    sample.gt_instances_3d.pop('track_id') # no need array
                    sample.gt_instances_3d.pop('bbox_3d_isvalid') # no need array
                    # sample.gt_instances_3d.pop('coop_isvalid') # no need array
                    sample.gt_instances_3d.pop('correlation') # no need array
                    sample.pred_instances_3d = pred_result[b]
                    import pdb

                    pdb.set_trace()
                    ret_list.append(sample)
                ################################ SHOW EGO SINGLE DETECT RESULT ################################
                # import os
                # os.makedirs('./data/step_vis_data', exist_ok=True)
                # visualizer: SimpleLocalVisualizer = SimpleLocalVisualizer.get_current_instance()
                # for idx, result in enumerate(ret_list):
                #     visualizer.set_points_from_npz(result.lidar_path)
                #     visualizer.draw_bev_bboxes(result.gt_instances_3d.bboxes_3d, c='#00FF00')
                #     thres = self.score_threshold
                #     result.pred_instances_3d = result.pred_instances_3d[result.pred_instances_3d['scores_3d'] > thres]
                #     visualizer.draw_bev_bboxes(result.pred_instances_3d.bboxes_3d, c='#FF0000')
                #     visualizer.just_save(f'./data/step_vis_data/single_result_{thres}_{self.ego_name}_{result.sample_idx}_{result.scene_name}.png')

                # import pdb
                # pdb.set_trace()
                ################################ SHOW EGO SINGLE DETECT RESULT ################################
            return ret_list