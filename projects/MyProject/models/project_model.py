from typing import Optional, Union, Dict, Sequence, List
from mmdet3d.registry import MODELS
from mmdet3d.models import MVXTwoStageDetector
from mmengine.device import get_device
import torch
from torch import Tensor
from ..utils import warp_features
import copy
from mmdet3d.structures import Det3DDataSample
from mmengine.structures import InstanceData
from ..visualization import SimpleLocalVisualizer
import numpy as np

@MODELS.register_module()
class ProjectModel(MVXTwoStageDetector):
    def __init__(self,
                 pts_voxel_encoder: Optional[dict] = None,
                 pts_middle_encoder: Optional[dict] = None,
                 pts_fusion_layer: Optional[dict] = None,
                 pts_backbone: Optional[dict] = None,
                 pts_neck: Optional[dict] = None,
                 temporal_backbone: Optional[dict] = None,
                 multi_task_head: Optional[dict] = None,
                 comm_expand_layer: Optional[dict] = None,
                 pts_train_cfg: Optional[dict] = None,
                 pts_test_cfg: Optional[dict] = None,
                 pts_fusion_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None):
        super(ProjectModel, self).__init__(
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_fusion_layer=pts_fusion_layer,
            pts_backbone=pts_backbone,
            pts_neck=pts_neck,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.pts_train_cfg = pts_train_cfg
        self.pts_test_cfg = pts_test_cfg
        self.pts_fusion_cfg = pts_fusion_cfg

        if temporal_backbone:
            self.temporal_backbone = MODELS.build(temporal_backbone)

        if multi_task_head:
            multi_task_head.update(train_cfg = pts_train_cfg)
            multi_task_head.update(test_cfg = pts_test_cfg)
            self.multi_task_head = MODELS.build(multi_task_head)
            self.fusion_multi_task_head = MODELS.build(multi_task_head)
        
        if comm_expand_layer:
            self.comm_expand_layer = MODELS.build(comm_expand_layer)

        if self.pts_train_cfg:
            self.gather_task_loss = self.pts_train_cfg.get('gather_task_loss', True) # type: ignore
            self.train_mode = self.pts_train_cfg.get('train_mode', 'single') # type: ignore
            assert self.train_mode in ('single', 'sparse_fusion', 'dense_fusion')

        if self.pts_test_cfg:
            self.test_mode = self.pts_test_cfg.get('test_mode', 'single') # type: ignore
            self.score_threshold = self.pts_test_cfg.get('score_threshold', 0.1)
            assert self.test_mode in ('full', 'where2comm', 'new_method', 'single')
        
        if self.pts_fusion_cfg:
            self.ego_name = self.pts_fusion_cfg.get("ego_name", "ego_vehicle")
            self.pc_range = self.pts_fusion_cfg.get("pc_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
            self.warp_size = (self.pc_range[3], self.pc_range[4])
        
        self.comm_rate = 0.0
        self.comm_count = 0

    @property
    def with_det_head(self):
        """bool: Whether the multi task head has a det head."""
        return hasattr(self,
                       'multi_task_head') and self.multi_task_head.det_head is not None

    @property
    def with_motion_head(self):
        """bool: Whether the multi task head has a motion head."""
        return hasattr(self,
                       'multi_task_head') and self.multi_task_head.motion_head is not None
    
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
        """
        scene_info: batch list of BaseDataElement
        example_seq: seq list - agent list - batch list of dict(inputs, data_samples) inputs[imgs, points, voxels] list[Tensor]
        or Tensor, data_samples: batch list of Det3DSamples

        SCENE_INFO

        'pose_matrix', 'sample_idx', 'future_motion_matrix', 'scene_timestamps', 'present_idx', 'scene_length', 'co_agents', 'co_length',
        'seq_timestamps', 'scene_name', 'seq_length'
        (str, array, int, float)

        DATA_SAMPLES:

        'vehicle_speed_x', 'sample_idx', 'bev_path', 'img_path', 'lidar_path', 'lidar2ego', 'lidar2cam', 'num_pts_feats', 'box_type_3d',
        'vehicle_speed_y', 'cam2img', 'ego2global', 'box_mode_3d'
        (str, int, float, list)
        
        'gt_instances', 'gt_pts_seg', 'eval_ann_info', 'gt_instances_3d'
        (None, Instance)

        """
        # sample_idx = scene_info[0].sample_idx
        # scene_name = scene_info[0].scene_name
        # scene_timestamps = scene_info[0].scene_timestamps
        # scene_length = scene_info[0].scene_length
        present_idx = scene_info[0].present_idx
        co_agents = scene_info[0].co_agents
        co_length = scene_info[0].co_length
        # seq_timestamps = scene_info[0].seq_timestamps
        # seq_length = scene_info[0].seq_length
        batch_size = len(scene_info)
        self.ego_id = co_agents.index(self.ego_name)
        assert mode in ('loss', 'predict')

        ################################ INPUT DEBUG (stop here)################################
        # scene_info[0].pop('pose_matrix')
        # scene_info[0].pop('future_motion_matrix')
        # print(scene_info[0])
        # import pdb
        # pdb.set_trace()
        # if mode == 'loss': 
        #     return {'fakeloss' : torch.ones(1, dtype=torch.float32, device=get_device(), requires_grad=True)}
        # else:
        #     return []
        ################################ INPUT DEBUG (stop here)################################

        # 0, 1, 2, ..., n so first[0] will be the earliest history, last[-1] is the future feature
        # Note that the entire sequence contains all frames from history, present, and future
        # so history + furure(including the present) = seq_length

        # only present inputs and labels matter
        agent_batch_features = []
        agent_batch_samples = []
        agent_batch_input_metas = []
        for j, agent in enumerate(co_agents):
            present_example_seq = example_seq[present_idx][j]
            present_batch_input_dict = present_example_seq['inputs']
            present_batch_input_meta = [present_example_seq['data_samples'][b].metainfo for b in range(batch_size)]

            agent_batch_samples.extend(present_example_seq['data_samples']) # A*B DSP
            agent_batch_input_metas.extend(present_batch_input_meta)

            pts_feat_dict = self.extract_feat(present_batch_input_dict,
                                                present_batch_input_meta,
                                                extract_level = 2,
                                                return_voxel_features = False,
                                                return_middle_features = True,
                                                return_backbone_features = False,
                                                return_neck_features = False)
            agent_batch_features.append(pts_feat_dict['middle_features']) # type: ignore
        agent_batch_features = torch.stack(agent_batch_features, dim=0) # A B C H W
        A, B, C, H, W = agent_batch_features.shape
        agent_batch_features = agent_batch_features.view(A*B, C, H, W)

        ################################ SHOW ORIGINAL PILLAR SCATTER ################################
        # import os
        # os.makedirs('./data/step_vis_data', exist_ok=True)
        # assert batch_size == 1
        # visualizer: SimpleLocalVisualizer = SimpleLocalVisualizer.get_current_instance()
        # for idx, feat in enumerate(agent_batch_features):
        #     visualizer.draw_bev_feat(feat)
        #     visualizer.just_save(f'./data/step_vis_data/scatter_feat_{co_agents[idx]}.png')
        
        # import pdb
        # pdb.set_trace()
        # if mode == 'loss': 
        #     return {'fakeloss' : torch.ones(1, dtype=torch.float32, device=get_device(), requires_grad=True)}
        # else:
        #     return []
        ################################ SHOW ORIGINAL PILLAR SCATTER ################################

        backbone_features = self.pts_backbone(agent_batch_features) # A*B C H W
        neck_features = self.pts_neck(backbone_features) # list A*B C H W

        agent_batch_visible_instances = []
        for samples in agent_batch_samples:
            valid_mask = samples.gt_instances_3d.bbox_3d_isvalid
            agent_batch_visible_instances.append(samples.gt_instances_3d[valid_mask]) # A*B visible

        ################################ DRAW VISIBLE TARGET ################################
        # import os
        # os.makedirs('./data/step_vis_data', exist_ok=True)
        # assert batch_size == 1
        # visualizer: SimpleLocalVisualizer = SimpleLocalVisualizer.get_current_instance()
        # for idx, visible_instance in enumerate(agent_batch_visible_instances):
        #     visualizer.set_points_from_npz(agent_batch_input_metas[idx]['lidar_path'])
        #     visualizer.draw_bev_bboxes(visible_instance.bboxes_3d, c='#00FF00')
        #     visualizer.just_save(f'./data/step_vis_data/visible_bboxes_{co_agents[idx]}.png')

        # import pdb
        # pdb.set_trace()
        # if mode == 'loss': 
        #     return {'fakeloss' : torch.ones(1, dtype=torch.float32, device=get_device(), requires_grad=True)}
        # else:
        #     return []
        ################################ DRAW VISIBLE TARGET ################################

        if (mode == 'loss' and self.train_mode != 'single') or mode == 'predict':
            ego_coop_samples = agent_batch_samples[self.ego_id*batch_size : (self.ego_id+1)*batch_size] # fetch ego coop bboxes
            ego_coop_input_metas = [samples.metainfo for samples in ego_coop_samples] # 1*B
            ego_coop_instances = [samples.gt_instances_3d for samples in ego_coop_samples] # 1*B
            for b in range(batch_size):
                ego_coop_instances[b].coop_isvalid = ego_coop_instances[b].bbox_3d_isvalid
            for j in range(co_length):
                if j != self.ego_id:
                    other_coop_instances = agent_batch_visible_instances[j*batch_size : (j+1)*batch_size] # fetch other coop bboxes, visible only
                    for b in range(batch_size):
                        ego_track_id = ego_coop_instances[b].track_id
                        # other visible 
                        other_track_id = other_coop_instances[b].track_id
                        in_mask = np.isin(ego_track_id, other_track_id)
                        new_isvalid = copy.deepcopy(ego_coop_instances[b].coop_isvalid)
                        new_isvalid[in_mask] = True
                        ego_coop_instances[b].coop_isvalid = new_isvalid # global valid bboxes for ego

            ################################ DRAW VISIBLE SINGLE AND COOP TARGET ################################
            # import os
            # os.makedirs('./data/step_vis_data', exist_ok=True)
            # assert batch_size == 1
            # visualizer: SimpleLocalVisualizer = SimpleLocalVisualizer.get_current_instance()
            # visualizer.set_points_from_npz(ego_coop_input_metas[0]['lidar_path'])
            # visualizer.draw_bev_bboxes(ego_coop_instances[0][ego_coop_instances[0].coop_isvalid].bboxes_3d, c='#FFFF00')
            # visualizer.draw_bev_bboxes(ego_coop_instances[0][ego_coop_instances[0].bbox_3d_isvalid].bboxes_3d, c='#00FF00')
            # visualizer.just_save(f'./data/step_vis_data/visible_single_coop_bboxes_{self.ego_name}.png')

            # import pdb
            # pdb.set_trace()
            # if mode == 'loss': 
            #     return {'fakeloss' : torch.ones(1, dtype=torch.float32, device=get_device(), requires_grad=True)}
            # else:
            #     return []
            ################################ DRAW VISIBLE SINGLE AND COOP TARGET ################################
                        
        if (mode == 'loss' and self.train_mode != 'single') or (mode == 'predict' and self.test_mode != 'single'):
            indices = [idx for idx in range(A)]
            indices.remove(self.ego_id)

            fusion_features = neck_features[0].view(A, B, *neck_features[0].shape[1:]) # A B C H W # FIXME single scale
            agent_fusion_features = fusion_features[indices] # A-1 B C H W
            ego_fusion_features = fusion_features[self.ego_id] # B C H W
        
        if mode == 'loss':
            single_head_feat_dict = self.multi_task_head(neck_features) # out from dethead and motionhead
            heatmaps, anno_boxes, inds, masks = self.multi_task_head.det_head.get_targets(agent_batch_visible_instances) # A*B
            # T * [A*B C H W] T * [A*B M 8/10] T * [A*B M] T * [A*B M] T * [A*B C H W]
            single_det_gt = {
                'heatmaps':heatmaps,
                'anno_boxes':anno_boxes,
                'inds':inds,
                'masks':masks,
            }
            single_det_loss_dict = self.multi_task_head.loss(single_head_feat_dict,
                                                det_gt = single_det_gt,
                                                motion_gt = None,
                                                gather_task_loss = self.gather_task_loss)
            if self.train_mode == 'single':
                return single_det_loss_dict
            
            elif self.train_mode == 'sparse_fusion':
                predict_dict = self.multi_task_head.predict(single_head_feat_dict,  agent_batch_input_metas, return_heatmaps=True)
                if 'det_pred' in predict_dict:
                    heatmaps = predict_dict['det_pred']
                    heatmaps = torch.cat(heatmaps, dim=1) # A*B c1+c2+c... H W
                    heatmaps = torch.max(heatmaps, dim=1, keepdim=True).values # AB 1 H W
                    comm_masks = torch.where(
                        heatmaps > self.score_threshold,
                        torch.ones_like(heatmaps, device=get_device()),
                        torch.zeros_like(heatmaps, device=get_device()),
                    ) # AB 1 H W

                    comm_masks = self.comm_expand_layer(comm_masks) # AB 1 H W # type: ignore
                ################################ SHOW SPARSE TRAIN (COMM) MASK ################################
                # import os
                # os.makedirs('./data/step_vis_data', exist_ok=True)
                # assert batch_size == 1
                # visualizer: SimpleLocalVisualizer = SimpleLocalVisualizer.get_current_instance()
                # for idx, feat in enumerate(comm_masks):
                #     visualizer.draw_bev_feat(feat)
                #     visualizer.just_save(f'./data/step_vis_data/sparse_train_mask_{co_agents[idx]}.png')
                
                # import pdb
                # pdb.set_trace()
                # return {'fakeloss' : torch.ones(1, dtype=torch.float32, device=get_device(), requires_grad=True)}
                ################################ SHOW SPARSE TRAIN (COMM) MASK ################################
                
            elif self.train_mode == 'dense_fusion':
                AB, _, H, W = heatmaps[0].shape
                comm_masks = torch.ones((AB, 1, H, W), device=get_device())
        
        elif mode == 'predict':
            if self.test_mode == 'single':
                ego_features = neck_features[0][self.ego_id * batch_size: (self.ego_id + 1)*batch_size] # 1*B C H W for single ego test
                if len(ego_features.shape) == 3:
                    ego_features = ego_features.unsqueeze(0)
                single_head_feat_dict = self.multi_task_head([ego_features]) # out from dethead and motionhead # FIXME only forward ego feature
                ego_input_metas = agent_batch_input_metas[self.ego_id * batch_size: (self.ego_id + 1)*batch_size] # 1*B
                ego_visible_instances = []
                for instance in ego_coop_instances: # type: ignore
                    ego_visible_instances.append(instance[instance.coop_isvalid])
                ret_list = []
                predict_dict = self.multi_task_head.predict(single_head_feat_dict,  ego_input_metas)
                if 'det_pred' in predict_dict:
                    pred_result = predict_dict['det_pred'] # add to pred_instances_3d from None to instance of bboxes_3d scores_3d labels_3d
                    for b in range(batch_size):
                        sample = Det3DDataSample()
                        sample.set_metainfo(
                            dict(
                                scene_sample_idx = scene_info[b].sample_idx,
                                scene_name = scene_info[b].scene_name,
                                agent_name = self.ego_name,
                                sample_idx = ego_input_metas[b]['sample_idx'], # type: ignore
                                box_type_3d = ego_input_metas[b]['box_type_3d'], # type: ignore
                                lidar_path = ego_input_metas[b]['lidar_path'], # type: ignore
                            )
                        )
                        sample.gt_instances_3d = ego_visible_instances[b] # type: ignore
                        sample.gt_instances_3d.pop('track_id') # no need array
                        sample.gt_instances_3d.pop('bbox_3d_isvalid') # no need array
                        sample.gt_instances_3d.pop('coop_isvalid') # no need array
                        sample.gt_instances_3d.pop('importance') # no need array
                        sample.pred_instances_3d = pred_result[b]
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
            
            elif self.test_mode == 'where2comm':
                single_head_feat_dict = self.multi_task_head(neck_features) # out from dethead and motionhead
                predict_dict = self.multi_task_head.predict(single_head_feat_dict,  agent_batch_input_metas, return_heatmaps=True)
                if 'det_pred' in predict_dict:
                    heatmaps = predict_dict['det_pred']
                    heatmaps = torch.cat(heatmaps, dim=1) # A*B c1+c2+c... H W
                    heatmaps = torch.max(heatmaps, dim=1, keepdim=True).values # AB 1 H W
                    comm_masks = torch.where(
                        heatmaps > self.score_threshold,
                        torch.ones_like(heatmaps, device=get_device()),
                        torch.zeros_like(heatmaps, device=get_device()),
                    ) # AB 1 H W

                    comm_masks = self.comm_expand_layer(comm_masks) # AB 1 H W # type: ignore
                    ################################ SHOW SPARSE TEST WHERE2COMM (COMM) MASK ################################
                    # import os
                    # os.makedirs('./data/step_vis_data', exist_ok=True)
                    # assert batch_size == 1
                    # visualizer: SimpleLocalVisualizer = SimpleLocalVisualizer.get_current_instance()
                    # for idx, feat in enumerate(comm_masks):
                    #     visualizer.draw_bev_feat(feat)
                    #     visualizer.just_save(f'./data/step_vis_data/sparse_test_where2comm_mask_{co_agents[idx]}.png')
                    
                    # import pdb
                    # pdb.set_trace()
                    # return []
                    ################################ SHOW SPARSE TEST WHERE2COMM (COMM) MASK ################################
                    
            elif self.test_mode == 'new_method':
                relamaps = self.multi_task_head.det_head.get_relamaps(agent_batch_visible_instances)
                relamaps = torch.stack(relamaps) # A*B c1+c2+c... 1 H W
                relamaps = torch.squeeze(relamaps, dim=2)# A*B c1+c2+c... H W
                relamaps = torch.max(relamaps, dim=1, keepdim=True).values # AB 1 H W
                comm_masks = torch.where(
                    relamaps > self.score_threshold,
                    torch.ones_like(relamaps, device=get_device()),
                    torch.zeros_like(relamaps, device=get_device()),
                ) # AB 1 H W

                comm_masks = self.comm_expand_layer(comm_masks) # AB 1 H W # type: ignore
                ################################ SHOW SPARSE TEST NEW_METHOD (COMM) MASK ################################
                # import os
                # os.makedirs('./data/step_vis_data', exist_ok=True)
                # assert batch_size == 1
                # visualizer: SimpleLocalVisualizer = SimpleLocalVisualizer.get_current_instance()
                # for idx, feat in enumerate(comm_masks):
                #     visualizer.draw_bev_feat(feat)
                #     visualizer.just_save(f'./data/step_vis_data/sparse_test_newmethod_mask_{co_agents[idx]}.png')
                
                # import pdb
                # pdb.set_trace()
                # return []
                ################################ SHOW SPARSE TEST NEW_METHOD (COMM) MASK ################################
            
            elif self.test_mode == 'full':
                AB, _, H, W = neck_features[0].shape # FIXME single scale
                comm_masks = torch.ones((AB, 1, H, W), device=get_device())

        fusion_masks = comm_masks.view(A, B, *comm_masks.shape[1:]) # A B 1 H W # type: ignore
        agent_fusion_masks = fusion_masks[indices] # A-1 B 1 H W # type: ignore
        # ego_fusion_masks = fusion_masks[self.ego_id] # B 1 H W

        if mode == 'predict' and self.test_mode != 'single':
            if self.test_mode == 'full':
                self.comm_rate = 1.0
                print(self.comm_rate)
            else:
                rate = torch.sum(agent_fusion_masks[0] > 0.5) / agent_fusion_masks[0].numel()
                self.comm_rate += rate.item()
                self.comm_count += 1
                print(self.comm_rate / self.comm_count)
        else:
            self.comm_rate = 0.0
            print(self.comm_rate)

        agent_comm_features = agent_fusion_features * agent_fusion_masks # A-1 B C H W # type: ignore
            
        agent_warpped_comm_features = []

        if mode == 'predict' and self.test_mode != 'single':
            batch_other_impo_instances = []

        for b in range(batch_size):
            present_pose_matrix = scene_info[b].pose_matrix[present_idx, indices, self.ego_id, ...] # use ego to other1, other2, ... # type: ignore
            agent_comm_feat = agent_comm_features[:, b] # A-1, C, H, W
            warp_comm_feat = warp_features(agent_comm_feat, present_pose_matrix, self.warp_size) # A-1 C H W
            agent_warpped_comm_features.append(warp_comm_feat)
            if mode == 'predict' and self.test_mode != 'single':
                inv_present_pose_matrix = scene_info[b].pose_matrix[present_idx, self.ego_id, indices, ...] # use other1, other2, ... to ego # type: ignore
                other_visible_instances = [agent_batch_visible_instances[idx*batch_size + b] for idx in indices] # type: ignore
                other_impo_instances = [instance[instance.importance] for instance in other_visible_instances] # importance A-1
                for idx, _ in enumerate(indices): #type: ignore
                    trans = inv_present_pose_matrix[idx]
                    other_impo_instances[idx].bboxes_3d.rotate(trans[:3, :3].T, None)
                    other_impo_instances[idx].bboxes_3d.translate(trans[:3, 3])
                batch_other_impo_instances.append(InstanceData.cat(other_impo_instances)) # type: ignore
        ################################ SHOW WARPPED FEATURE ################################
        # import os
        # os.makedirs('./data/step_vis_data', exist_ok=True)
        # assert batch_size == 1
        # visualizer: SimpleLocalVisualizer = SimpleLocalVisualizer.get_current_instance()
        # visualizer.draw_bev_feat(ego_fusion_features[0]) # type: ignore
        # visualizer.just_save(f'./data/step_vis_data/ego_warp_feat_{self.ego_id}')
        # for idx, feat in enumerate(agent_warpped_comm_features[0]):
        #     visualizer.draw_bev_feat(feat)
        #     visualizer.just_save(f'./data/step_vis_data/other_agent_warp_feat_{idx}')
        
        # import pdb
        # pdb.set_trace()
        # if mode == 'loss': 
        #     return {'fakeloss' : torch.ones(1, dtype=torch.float32, device=get_device(), requires_grad=True)}
        # else:
        #     return []
        ################################ SHOW WARPPED FEATURE ################################
        agent_warpped_comm_features = torch.stack(agent_warpped_comm_features, dim=0) # B A-1 C H W
        # coop send and receive
        ego_fusion_features = ego_fusion_features.unsqueeze(1) # B 1 C H W # type: ignore
        ego_fusion_result = self.pts_fusion_layer(ego_fusion_features, agent_warpped_comm_features).squeeze(1) # B C H W
        coop_head_feat_dict = self.multi_task_head([ego_fusion_result]) # out from dethead and motionhead

        coop_instances = []
        for instance in ego_coop_instances: # type: ignore
            coop_instances.append(instance[instance.coop_isvalid])
        
        if mode == 'loss':
            loss_dict = {}
            HM, AB, ID, MS = self.multi_task_head.det_head.get_targets(coop_instances) # A*B final fusion detect # type: ignore
            # T * [A*B C H W] T * [A*B M 8/10] T * [A*B M] T * [A*B M]
            coop_det_gt = {
                'heatmaps':HM,
                'anno_boxes':AB,
                'inds':ID,
                'masks':MS,
            }
            coop_det_loss_dict = self.multi_task_head.loss(coop_head_feat_dict,
                                                det_gt = coop_det_gt,
                                                motion_gt = None,
                                                gather_task_loss = self.gather_task_loss)
            
            for k, v in single_det_loss_dict.items(): # type: ignore
                loss_dict[f"single_{k}"] = v
            for k, v in coop_det_loss_dict.items():
                loss_dict[f"coop_{k}"] = v

            return loss_dict

        elif mode == 'predict':
            ret_list = []
            predict_dict = self.multi_task_head.predict(coop_head_feat_dict,  ego_coop_input_metas) # type: ignore

            ego_visible_instances = []
            for instance in ego_coop_instances: # type: ignore
                ego_visible_instances.append(instance[instance.bbox_3d_isvalid])

            if 'det_pred' in predict_dict:
                pred_result = predict_dict['det_pred'] # add to pred_instances_3d from None to instance of bboxes_3d scores_3d labels_3d
                for b in range(batch_size):
                    sample = Det3DDataSample()
                    sample.set_metainfo(
                        dict(
                            scene_sample_idx = scene_info[b].sample_idx,
                            scene_name = scene_info[b].scene_name,
                            agent_name = self.ego_name,
                            sample_idx = ego_coop_input_metas[b]['sample_idx'], # type: ignore
                            box_type_3d = ego_coop_input_metas[b]['box_type_3d'], # type: ignore
                            lidar_path = ego_coop_input_metas[b]['lidar_path'], # type: ignore
                        )
                    )


                    # 用于经过new_method的评估
                    # impo_track_id = batch_other_impo_instances[b].track_id # type: ignore
                    # ego_track_id = ego_visible_instances[b].track_id
                    # in_mask = np.isin(ego_track_id, impo_track_id)
                    # notin_mask = np.logical_not(in_mask)
                    # in_1 = ego_visible_instances[b][notin_mask]
                    # in_1.pop('track_id') # no need array
                    # in_1.pop('bbox_3d_isvalid') # no need array
                    # in_1.pop('coop_isvalid') # no need array
                    # in_1.pop('importance') # no need array
                    # in_2 = batch_other_impo_instances[b] # type: ignore
                    # in_2.pop('track_id') # no need array
                    # in_2.pop('bbox_3d_isvalid') # no need array
                    # in_2.pop('importance') # no need array
                    # ins = [in_1, in_2]
                    # sample.gt_instances_3d = InstanceData.cat(ins)

                    # 用于协同评估
                    sample.gt_instances_3d = coop_instances[b] # type: ignore
                    sample.gt_instances_3d.pop('track_id') # no need array
                    sample.gt_instances_3d.pop('bbox_3d_isvalid') # no need array
                    sample.gt_instances_3d.pop('coop_isvalid') # no need array
                    sample.gt_instances_3d.pop('importance') # no need array


                    sample.pred_instances_3d = pred_result[b]
                    ret_list.append(sample)
                ################################ SHOW EGO SINGLE DETECT RESULT ################################
                import os
                os.makedirs('./data/step_vis_data', exist_ok=True)
                visualizer: SimpleLocalVisualizer = SimpleLocalVisualizer.get_current_instance()
                coop_instances_ori = agent_batch_visible_instances[self.ego_id * batch_size: (self.ego_id + 1)*batch_size]
                for idx, result in enumerate(ret_list):
                    visualizer.set_points_from_npz(result.lidar_path)


                    # 用于协同的可视化
                    visualizer.draw_bev_bboxes(result.gt_instances_3d.bboxes_3d, c='#FFFF00')
                    visualizer.draw_bev_bboxes(coop_instances_ori[idx].bboxes_3d, c='#00FF00')

                    # 用于经过new_method的可视化
                    # visualizer.draw_bev_bboxes(result.gt_instances_3d.bboxes_3d, c='#00FF00')


                    visualizer.draw_bev_bboxes(batch_other_impo_instances[idx].bboxes_3d, c='#00BFFF') # type: ignore
                    thres = self.score_threshold
                    result.pred_instances_3d = result.pred_instances_3d[result.pred_instances_3d['scores_3d'] > thres]
                    visualizer.draw_bev_bboxes(result.pred_instances_3d.bboxes_3d, c='#FF0000')
                    visualizer.just_save(f'./data/step_vis_data/coop_result_{thres}_{self.ego_name}_{result.sample_idx}_{result.scene_name}.png')

                import pdb
                pdb.set_trace()
                ################################ SHOW EGO SINGLE DETECT RESULT ################################
            return ret_list # type: ignore
        
        else:
            return neck_features[0] # FIXME single scale