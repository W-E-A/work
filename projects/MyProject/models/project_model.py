from typing import Optional, Union, Dict, Tuple, Sequence, List
from mmdet3d.registry import MODELS
from mmdet3d.models import MVXTwoStageDetector
from mmengine.structures import InstanceData
from mmengine.device import get_device
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from mmdet3d.models.utils import (clip_sigmoid, draw_heatmap_gaussian, gaussian_radius)
from mmdet3d.models.layers import circle_nms
from ..utils import calc_relative_pose, simple_points_project

@MODELS.register_module()
class ProjectModel(MVXTwoStageDetector):
    def __init__(self,
                 pts_voxel_encoder: Optional[dict] = None,
                 pts_middle_encoder: Optional[dict] = None,
                 pts_backbone: Optional[dict] = None,
                 pts_neck: Optional[dict] = None,
                 temporal_backbone: Optional[dict] = None,
                 multi_task_head: Optional[dict] = None,
                 pts_train_cfg: Optional[dict] = None,
                 pts_test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None):
        super(ProjectModel, self).__init__(
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_backbone=pts_backbone,
            pts_neck=pts_neck,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.pts_train_cfg = pts_train_cfg
        self.pts_test_cfg = pts_test_cfg

        if temporal_backbone:
            self.temporal_backbone = MODELS.build(temporal_backbone)

        if multi_task_head:
            multi_task_head.update(train_cfg = pts_train_cfg)
            multi_task_head.update(test_cfg = pts_test_cfg)
            self.multi_task_head = MODELS.build(multi_task_head)

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
        if self.with_voxel_encoder:
            voxel_features = self.pts_voxel_encoder(voxel_dict['voxels'], # [n*bs, N, feat]
                                                    voxel_dict['num_points'], # [n*bs, ] realpoints
                                                    voxel_dict['coors'], # [n*bs, 1 + 3] batch z y x
                                                    img_feats,
                                                    batch_input_metas)
            if return_voxel_features:
                return_dict['voxel_features'] = voxel_features
            if extract_level == 1:
                return return_dict
        if self.with_middle_encoder and self.with_voxel_encoder:
            if len(voxel_dict['coors'].shape) < 4:
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
        if self.with_pts_backbone and self.with_middle_encoder:
            # [bs, feat, 1024, 1024] [N, C, H, W]
            backbone_features = self.pts_backbone(middle_features) # return tensor or sequence # type: ignore
            if return_backbone_features:
                return_dict['backbone_features'] = backbone_features
            if extract_level == 3:
                return return_dict
        if self.with_pts_neck and self.with_pts_backbone:
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

        DATA_SAMPLES:

        'cam2img', 'num_pts_feats', 'lidar_path', 'img_path', 'box_mode_3d', 'vehicle_speed_y',
        'lidar2cam', 'vehicle_speed_x', 'lidar2ego', 'ego2global', 'box_type_3d', 'sample_idx', 'bev_path'
        
        'gt_instances', 'gt_pts_seg', 'gt_instances_3d', 'eval_ann_info'

        """
        
        assert mode in ('tensor', 'loss', 'predict', 'coop', 'coop_predict')

        co_length = scene_info[0].co_length
        co_agents = scene_info[0].co_agents
        seq_length = scene_info[0].seq_length
        scene_length = scene_info[0].scene_length
        present_idx = scene_info[0].present_idx
        batch_size = len(scene_info)
        print(scene_info[0])

        #############################################################################
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from matplotlib.cm import get_cmap
        # seq_length in exp1 must be a very large positive int, and use adaptive seq length settings.
        assert seq_length == scene_length

        cmaps = [
            'Greys', 'Purples', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
        ]

        grid_size = self.pts_train_cfg.get('grid_size', None) # type: ignore
        voxel_size = self.pts_train_cfg.get('voxel_size', None) # type: ignore
        # point_cloud_range = self.pts_train_cfg.get('point_cloud_range', None) # type: ignore
        # out_size_factor = self.pts_train_cfg.get('out_size_factor', None) # type: ignore
        # offset_x = point_cloud_range[0] + voxel_size[0] * 0.5
        # offset_y = point_cloud_range[1] + voxel_size[1] * 0.5

        predict_size = 8 # 4 second for motion prediction
        seq_start_from = 0 # ego motion start timestamp
        random_start = True
        distance_thres = 30 # 20m for candidates
        interrupt_thres = 5 # 5m for potential interrupts

        if random_start:
            seq_start_from = np.random.randint(0, seq_length - predict_size + 1)
        else:
            seq_start_from = min(max(seq_start_from, 0), seq_length - predict_size)

        track_map = {}
        
        ego_pose = [] # global
        for i in range(seq_start_from, seq_length):
            data_samples = example_seq[i][0]['data_samples'][0]

            bev_path = data_samples.metainfo['bev_path']
            if i == seq_start_from:
                bev_insmap = np.load(bev_path)['data'] # npz
                H, W, _ = bev_insmap.shape # type: ignore
                h, w, _ = tuple(grid_size)
                minx = min(max((W - w) // 2 - 1, 0), grid_size[1] - 1)
                miny = min(max((H - h) // 2 - 1, 0), grid_size[0] - 1)
                maxx = max(min(w + (W - w) // 2, grid_size[1]), 1)
                maxy = max(min(h + (H - h) // 2, grid_size[0]), 1)
                bev_insmap = bev_insmap[miny:maxy,minx:maxx] # type: ignore

            ego_frame_metainfo = data_samples.metainfo
            ego2global = torch.tensor(ego_frame_metainfo['ego2global'], dtype=torch.float32, device=get_device())
            lidar2ego = torch.tensor(ego_frame_metainfo['lidar2ego'], dtype=torch.float32, device=get_device())
            lidar2global = ego2global @ lidar2ego # 4 4

            lidar2global_rotation_T = lidar2global[:3, :3].T
            lidar2global_translation = lidar2global[:3, 3]
            data_samples.gt_instances_3d.bboxes_3d.rotate(lidar2global_rotation_T, None)
            data_samples.gt_instances_3d.bboxes_3d.translate(lidar2global_translation)

            # gt_boxes_center_lidar = data_samples.gt_instances_3d.bboxes_3d.gravity_center.unsqueeze(0) # 1 N 3
            # gt_boxes_center_global = simple_points_project(gt_boxes_center_lidar, lidar2global)# 1 N 3

            instances = data_samples.gt_instances_3d
            if i == seq_start_from: # 第一帧雷达可视范围内检测到了， 之后用的真值认为是预测结果
                instances = instances[instances.bbox_3d_isvalid == False] # type: ignore
            track_id = instances.track_id
            for id in track_id:
                if i == seq_start_from: # 启动第一帧跟踪所有可视的对象的id
                    track_map[id.item()] = {
                        'cmap': cmaps[np.random.randint(0, len(cmaps))],
                        'instance': instances[track_id == id.item()],
                    }
                elif id.item() in track_map:
                    if len(track_map[id.item()]['instance']) >= predict_size: # 对于跟踪大于预测之后停止跟踪，认为是当前的motion预测的值
                        continue
                    track_map[id.item()]['instance'] = InstanceData.cat([track_map[id.item()]['instance'], instances[track_id == id.item()]])

            ego_pose.append(ego2global) # type: ignore
        
        scatter_trans = torch.tensor(
            [[0, 1],
             [-1, 0]],
             dtype=torch.float32,
             device=get_device()
        )
        ego_pose_base = ego_pose[0] # fake world，基于第一帧检测起点作为EGO当前帧的所有目标的基准世界位置
        ego_pose_rela = calc_relative_pose(ego_pose_base, ego_pose)
        ego_pose_rela_center = torch.stack([pose[:2, 3] for pose in ego_pose_rela]) # [7, 2] EGO 规划真值位置序列
        ego_start_center = ego_pose_rela_center[0]

        ego_pose_rela_center_vis = ego_pose_rela_center @ scatter_trans.T # EGO 规划平面位置序列
        ego_pose_rela_x_vis = torch.round(ego_pose_rela_center_vis[:, 0] / voxel_size[0] + grid_size[1] * 0.5).int().cpu().numpy()
        ego_pose_rela_y_vis = torch.round(ego_pose_rela_center_vis[:, 1] / voxel_size[1] + grid_size[0] * 0.5).int().cpu().numpy()
        distance_radius = distance_thres / voxel_size[0] # FIXME
        interrupt_radius = interrupt_thres / voxel_size[0] # FIXME
        
        fig, ax = plt.subplots(1, 1)
        ax.imshow(bev_insmap) # type: ignore
        ego_cmap = get_cmap('Blues')
        c = np.linspace(0.0, 1.0, seq_length - seq_start_from)[::-1]
        ax.scatter(ego_pose_rela_x_vis, ego_pose_rela_y_vis, c=c, cmap='Blues', s=6)
        circle = Circle((ego_pose_rela_x_vis[0], ego_pose_rela_y_vis[0]), distance_radius, edgecolor='cyan', fill=False, linewidth=0.5) # 第一帧的匹配范围绘制
        circles = [Circle((ego_pose_rela_x_vis[i], ego_pose_rela_y_vis[i]), interrupt_radius, edgecolor=ego_cmap(c[i]), fill=False, linewidth=0.25) for i in range(seq_length - seq_start_from)]
        circles.append(circle)
        for circle in circles:
            ax.add_patch(circle)

        for i, seq in enumerate(list(range(seq_start_from, seq_length))): # 绘制从seq_start_from开始到结束的EGO驾驶规划意图
            ax.text(ego_pose_rela_x_vis[i], ego_pose_rela_y_vis[i], s=str(seq), fontsize=3, ha='center', va='center')
            ax.text(ego_pose_rela_x_vis[i], ego_pose_rela_y_vis[i], s="  EGO", fontsize=3, ha='left', va='center')

        # 开始筛选目标，判断潜在的碰撞风险
        for track_id, v in track_map.items():
            cmap = v['cmap']
            instance = v['instance']
            last = len(instance)
            if last < predict_size: # 预测长度不足predict_size的跟踪轨迹直接忽略（实际motion输出一定是定长的）
                continue
            label = instance.labels_3d[0]
            
            gt_boxes_xyz_global = instance.bboxes_3d.gravity_center.unsqueeze(0) # [1, N, 3] 基于真实世界的位置需要转换到相对于基准位置
            gt_boxes_xyz_rela = simple_points_project(gt_boxes_xyz_global, torch.linalg.inv(ego_pose_base)).squeeze(0) # [N, 3] moiton 预测真值位置序列
            gt_boxes_center_rela = gt_boxes_xyz_rela[:, :2]
            start_center = gt_boxes_center_rela[0]
            if torch.norm(start_center - ego_start_center) > distance_thres: # 第一帧检测范围内的目标作为潜在候选
                continue
            valid = False
            interrupt_seq = []
            interrupt_i = []
            for i, seq in enumerate(list(range(seq_start_from, seq_start_from + predict_size))):
                center = gt_boxes_center_rela[i] # 这里是跟踪的固定长度
                ego_center = ego_pose_rela_center[i] # 这里必须从整个轨迹序列中开始取得
                if torch.norm(center - ego_center) <= interrupt_thres:
                    valid = True
                    interrupt_seq.append(seq)
                    interrupt_i.append(i)

            gt_boxes_center_vis = gt_boxes_center_rela @ scatter_trans.T # [N, 2] moiton 预测平面位置序列
            gt_boxes_x_vis = torch.round(gt_boxes_center_vis[:, 0] / voxel_size[0] + grid_size[1] * 0.5).int().cpu().numpy()
            gt_boxes_y_vis = torch.round(gt_boxes_center_vis[:, 1] / voxel_size[1] + grid_size[0] * 0.5).int().cpu().numpy()
            
            if not valid:
                ax.scatter(gt_boxes_x_vis, gt_boxes_y_vis, c='black', s=1)
                for i, seq in enumerate(list(range(seq_start_from, seq_start_from + predict_size))):
                    ax.text(gt_boxes_x_vis[i], gt_boxes_y_vis[i], s="  " + str(seq), fontsize=2.0, ha='left', va='top')
                    ax.text(gt_boxes_x_vis[i], gt_boxes_y_vis[i], s="  " + str(track_id), fontsize=2.0, ha='left', va='bottom')
            else:
                ax.scatter(gt_boxes_x_vis, gt_boxes_y_vis, c=np.arange(predict_size)[::-1], cmap=cmap, s=2)
                for i, seq in enumerate(list(range(seq_start_from, seq_start_from + predict_size))):
                    ax.text(gt_boxes_x_vis[i], gt_boxes_y_vis[i], s="  " + str(seq), fontsize=2.0, ha='left', va='top')
                    ax.text(gt_boxes_x_vis[i], gt_boxes_y_vis[i], s="  " + str(track_id), fontsize=2.0, ha='left', va='bottom')
                    if i in interrupt_i:
                        ax.scatter(gt_boxes_x_vis[i], gt_boxes_y_vis[i], marker='o', facecolors='none', edgecolors='blue', s=20, linewidths= 0.5)
        fig.savefig('./motion.png', dpi = 800)
        
        import pdb
        pdb.set_trace()

        return {'fakeloss' : torch.ones(1, dtype=torch.float32, device=get_device(), requires_grad=True)}

        #############################################################################


        # if mode == 'loss':
        #     scene_loss = [ [ {} for j in range(co_length)] for i in range(seq_length)] # task list of loss dict
        # elif mode == 'predict':
        #     scene_ret = [ [ [] for j in range(co_length)] for i in range(seq_length)] # task list of ret dict

        # 0, 1, 2, ..., n so first[0] will be the earliest history, last[-1] is the future feature
        # Note that the entire sequence contains all frames from history, present, and future
        # so history(including the present) + furure = seq_length
        history_seq_timestamps = list(range(seq_length))[:present_idx] # 2 can't be 0 at least 1
        history_length = len(history_seq_timestamps)
        future_seq_timestamps = list(range(seq_length))[present_idx:] # 1 # can be 0
        future_length = len(future_seq_timestamps)
        # assume seq_length == 3, future 1 history(present) 2

        raw_features = []
        for i in history_seq_timestamps:
            agent_features = []
            for j in range(co_length):

                batch_inputs_dict = example_seq[i][j]['inputs'] # ['inputs', 'single_samples', 'coop_samples]

                batch_single_samples = example_seq[i][j]['single_samples'] # data sample list
                batch_single_input_metas = [batch_single_samples[b].metainfo for b in range(batch_size)]
                # batch_single_instances_3d = [batch_single_samples[b].gt_instances_3d for b in range(batch_size)]

                # batch_coop_samples = example_seq[i][j]['coop_samples'] # data sample list
                # batch_coop_input_metas = [batch_coop_samples[b].metainfo for b in range(batch_size)]
                # batch_coop_instances = [batch_coop_samples[b].gt_instances_3d for b in range(batch_size)]

                pst_feat_dict = self.extract_feat(batch_inputs_dict,
                                                  batch_single_input_metas,
                                                  extract_level = 2,
                                                  return_voxel_features = False,
                                                  return_middle_features = True,
                                                  return_backbone_features = False,
                                                  return_neck_features = False)
                agent_features.append(pst_feat_dict['middle_features']) # type: ignore
            agent_features = torch.stack(agent_features, dim=0)
            raw_features.append(agent_features)
        raw_features = torch.stack(raw_features, dim=0) # S A B C H W
        S,A,B,C,H,W = raw_features.shape
        raw_features = raw_features.permute(2, 1, 0, 3, 4, 5).contiguous().view(B*A*S, C, H, W) # B*A*S C H W
        
        if self.with_pts_backbone:
            backbone_features = self.pts_backbone(raw_features)
        if self.with_pts_neck and self.with_pts_backbone:
            neck_features = self.pts_neck(backbone_features) # type: ignore
            # B*A*S C H W
            neck_features = neck_features.view(B*A, S, *neck_features.shape[1:])
            # B*A S C H W
        if self.with_motion_head and self.with_pts_neck:
            temporal_features = self.temporal_backbone(neck_features) # type: ignore
            # B*A C H W FIXME

                # head_feat_dict = self.multi_task_head(pst_feat_dict['neck_features']) # type: ignore

                # if mode == 'loss':
                #     loss_dict = {}
                #     gather_task_loss = self.pts_train_cfg.get('gather_task_loss', True) # type: ignore
                #     loss_dict = self.multi_task_head.loss(head_feat_dict,
                #                                         batch_det_instances = batch_single_instances_3d,
                #                                     #   batch_motion_instances = batch_single_instances_motion,
                #                                         batch_motion_instances = None,
                #                                         gather_task_loss = gather_task_loss)
                #     scene_loss[i][j] = loss_dict # type: ignore

                # elif mode == 'predict':
                #     predict_dict = self.multi_task_head.predict(head_feat_dict, batch_single_input_metas)
                #     if 'det_pred' in predict_dict:
                #         example_seq[i][j]['single_samples'] = self.add_pred_to_datasample(batch_single_samples,
                #                                                                           predict_dict['det_pred'],
                #                                                                           None)
                #     # if 'motion_pred' in predict_dict:
                #     #     pass



        if mode == 'loss':
            # loss = {}
            # for i in range(seq_length):
            #     for j in range(co_length):
            #         prefix = f'S{i}_A{j}'
            #         for k, v in scene_loss[i][j].items(): # type: ignore
            #             loss[f'{prefix}_{k}'] = v
            # return loss
            return []
        elif mode == 'predict':
            return []
        else:
            return []
        
    
        
        
        