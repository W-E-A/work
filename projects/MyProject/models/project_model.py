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
from itertools import chain
import copy
from mmdet3d.structures import Det3DDataSample

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

        if self.pts_train_cfg:
            self.gather_task_loss = self.pts_train_cfg.get('gather_task_loss', True) # type: ignore


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

        'pose_matrix', 'sample_idx', 'motion_matrix', 'scene_timestamps', 'present_idx', 'scene_length', 'co_agents', 'co_length',
        'seq_timestamps', 'scene_name', 'seq_length'
        (str, array, int, float)

        DATA_SAMPLES:

        'vehicle_speed_x', 'sample_idx', 'bev_path', 'img_path', 'lidar_path', 'lidar2ego', 'lidar2cam', 'num_pts_feats', 'box_type_3d',
        'vehicle_speed_y', 'cam2img', 'ego2global', 'box_mode_3d'
        (str, int, float, list)
        
        'gt_instances', 'gt_pts_seg', 'eval_ann_info', 'gt_instances_3d'
        (None, Instance)

        """
        sample_idx = scene_info[0].sample_idx
        scene_name = scene_info[0].scene_name
        scene_timestamps = scene_info[0].scene_timestamps
        scene_length = scene_info[0].scene_length
        present_idx = scene_info[0].present_idx
        co_agents = scene_info[0].co_agents
        co_length = scene_info[0].co_length
        seq_timestamps = scene_info[0].seq_timestamps
        seq_length = scene_info[0].seq_length
        batch_size = len(scene_info)

        ################################ DEBUG ################################
        # scene_info[0]['pose_matrix'] = []
        # scene_info[0]['motion_matrix'] = []
        # print(scene_info[0])
        # import pdb
        # pdb.set_trace()
        # return {'fakeloss' : torch.ones(1, dtype=torch.float32, device=get_device(), requires_grad=True)}
        ################################ DEBUG ################################


        # if self.pts_train_cfg.get('visualize', False) and self.pts_train_cfg.get('vis_cfg', None) and mode == 'loss': # type: ignore
        #     self.train_visualize(scene_info, example_seq)
        #     return {'fakeloss' : torch.ones(1, dtype=torch.float32, device=get_device(), requires_grad=True)}

        # 0, 1, 2, ..., n so first[0] will be the earliest history, last[-1] is the future feature
        # Note that the entire sequence contains all frames from history, present, and future
        # so history + furure(including the present) = seq_length


        # only present inputs and labels matter
        agent_features = []
        agent_batch_samples = []
        for j, agnet in enumerate(co_agents):
            present_example_seq = example_seq[present_idx][j]
            present_batch_input_dict = present_example_seq['inputs']
            present_batch_input_meta = [present_example_seq['data_samples'][b].metainfo for b in range(batch_size)]

            agent_batch_samples.extend(present_example_seq['data_samples']) # A*B DSP

            pts_feat_dict = self.extract_feat(present_batch_input_dict,
                                                present_batch_input_meta,
                                                extract_level = 2,
                                                return_voxel_features = False,
                                                return_middle_features = True,
                                                return_backbone_features = False,
                                                return_neck_features = False)
            agent_features.append(pts_feat_dict['middle_features']) # type: ignore
        agent_features = torch.stack(agent_features, dim=0) # A B C H W
        A, B, C, H, W = agent_features.shape
        agent_features = agent_features.view(A*B, C, H, W)

        ################################ SHOW ORIGINAL PILLAR SCATTER ################################
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 1)
        # assert batch_size == 1

        # infra_feat = agent_features[1, ...].permute(1, 2, 0)
        # img = torch.sigmoid(torch.mean(infra_feat, dim=-1)).detach().cpu().numpy()
        # ax.imshow(img)
        # fig.savefig('./feat.png', dpi=300)
        
        # import pdb
        # pdb.set_trace()
        # return {'fakeloss' : torch.ones(1, dtype=torch.float32, device=get_device(), requires_grad=True)}
        ################################ SHOW ORIGINAL PILLAR SCATTER ################################

        backbone_features = self.pts_backbone(agent_features) # A*B C H W
        neck_features = self.pts_neck(backbone_features) # list A*B C H W
             
        head_feat_dict = self.multi_task_head(neck_features) # out from dethead and motionhead

        if mode == 'loss':
            loss_dict = {}
            all_visible_instances = []
            temp_samples = copy.deepcopy(agent_batch_samples)
            for samples in temp_samples:
                valid_mask = samples.gt_instances_3d.bbox_3d_isvalid
                all_visible_instances.append(samples.gt_instances_3d[valid_mask]) # A*B visible

            if self.with_det_head:
                heatmaps, anno_boxes, inds, masks, relamaps = self.multi_task_head.det_head.get_targets(all_visible_instances) # A*B
                # T * [A*B C H W] T * [A*B M 8/10] T * [A*B M] T * [A*B M]
                relamaps = torch.stack(relamaps, dim=0).permute(1, 0, 2, 3, 4).contiguous() # T A*B C H W -> A*B T C H W
                AB, T, C, H, W = relamaps.shape
                relamaps = relamaps.view(AB, T*C, H, W)
                relamaps = torch.max(relamaps, dim=1, keepdim=True).values # AB 1 H W
                relamasks = torch.where(
                    relamaps > 1e-8,
                    torch.ones_like(relamaps, device=get_device()),
                    torch.zeros_like(relamaps, device=get_device()),
                ) # AB 1 H W

                ################################ SHOW RELA(COMM) MASK ################################
                # import matplotlib.pyplot as plt
                # fig, (ax1, ax2) = plt.subplots(1, 2)
                # assert batch_size == 1

                # relamasks = relamasks.permute(0, 2, 3, 1)
                # ego_mask = relamasks[0].cpu().numpy().squeeze(-1)
                # infra_mask = relamasks[1].cpu().numpy().squeeze(-1)
                # ax1.imshow(ego_mask)
                # ax2.imshow(infra_mask)
                # fig.savefig('./mask.png', dpi=300)

                # import pdb
                # pdb.set_trace()
                # return {'fakeloss' : torch.ones(1, dtype=torch.float32, device=get_device(), requires_grad=True)}
                ################################ SHOW RELA(COMM) MASK ################################

                single_det_gt = {
                    'heatmaps':heatmaps,
                    'anno_boxes':anno_boxes,
                    'inds':inds,
                    'masks':masks,
                }

                loss_dict = self.multi_task_head.loss(head_feat_dict,
                                                    det_gt = single_det_gt,
                                                    motion_gt = None,
                                                    gather_task_loss = self.gather_task_loss)


                # comm_features = neck_features[0] * relamasks # A*B C H W * A*B 1 H W
                # _, C, H, W = comm_features.shape
                # # FIXME we assume 0 position will always be ego
                # agent_comm_features = comm_features.view(A, B, C, H, W)[1:] # A-1 B C H W

                # FIXME
                # FIXME 基于所有可视的GT， 对于EGO来说， 需要：协同代理可视GT -> 位姿变换，id筛选，拼接到EGO的感知范围 （多）（融合后监督）OLD!
                # FIXME 基于所有可视的GT， 对于EGO来说， 需要：策略筛选的GT -> 位姿变换，id筛选，拼接到EGO的感知范围 （少）（融合后监督）NEW!
                # FIXME
                # FIXME wrap the feature
                # FIXME into fusion layer
                # FIXME 获取协同标签
                # FIXME 进行协同监督
            
            return loss_dict

        if mode == 'predict':
            temp_samples = copy.deepcopy(agent_batch_samples)
            present_batch_input_metas = [samples.metainfo for samples in temp_samples] # A*B
            predict_dict = self.multi_task_head.predict(head_feat_dict, present_batch_input_metas)
            if 'det_pred' in predict_dict:
                result_list = predict_dict['det_pred'] # add to pred_instances_3d from None to instance of bboxes_3d scores_3d labels_3d
                ret_list = []
                for j, agent in enumerate(co_agents):
                    agent_pred_result = result_list[ j*batch_size : (j+1)*batch_size ]
                    agent_sample = temp_samples[ j*batch_size : (j+1)*batch_size ]
                    result_sample = []
                    for b in range(batch_size):
                        sample = Det3DDataSample()
                        sample.set_metainfo(
                            dict(
                                scene_sample_idx = scene_info[b].sample_idx,
                                scene_name = scene_info[b].scene_name,
                                agent_name = agent,
                                sample_idx = agent_sample[b].metainfo['sample_idx'], # type: ignore
                                box_type_3d = agent_sample[b].metainfo['box_type_3d'], # type: ignore
                            )
                        )
                        sample.gt_instances_3d = agent_sample[b].gt_instances_3d
                        valid_mask = sample.gt_instances_3d.bbox_3d_isvalid
                        sample.gt_instances_3d = sample.gt_instances_3d[valid_mask]
                        sample.gt_instances_3d.pop('track_id')
                        sample.gt_instances_3d.pop('bbox_3d_isvalid')
                        sample.pred_instances_3d = agent_pred_result[b]
                        result_sample.append(sample)
                    ret_list.extend(result_sample)

            # if 'motion_pred' in predict_dict:
            #     pass FIXME

            return ret_list # type: ignore
        
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
    
    # def train_visualize(self,
    #             scene_info: Sequence,
    #             example_seq: Sequence) -> None:
        
    #     co_length = scene_info[0].co_length
    #     co_agents = scene_info[0].co_agents
    #     seq_length = scene_info[0].seq_length
    #     seq_timestamps = scene_info[0].seq_timestamps
    #     scene_length = scene_info[0].scene_length
    #     present_idx = scene_info[0].present_idx
    #     batch_size = len(scene_info)
        
    #     import matplotlib.pyplot as plt
    #     from matplotlib.patches import Circle
    #     from matplotlib.cm import get_cmap
    #     # seq_length in exp1 must be a very large positive int, and use adaptive seq length settings.
    #     assert seq_length == scene_length

    #     cmaps = [
    #         'Greys', 'Purples', 'Greens', 'Oranges', 'Reds',
    #         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
    #     ]

    #     grid_size = self.pts_train_cfg.get('grid_size', None) # type: ignore
    #     voxel_size = self.pts_train_cfg.get('voxel_size', None) # type: ignore

    #     vis_cfg = self.pts_train_cfg.get('vis_cfg', None) # type: ignore
    #     assert vis_cfg != None

    #     predict_size = vis_cfg.get('predict_size', 6) # 4 second for motion prediction
    #     seq_start_from = vis_cfg.get('seq_start_from', 5) # ego motion start timestamp
    #     random_start = vis_cfg.get('random_start', False) # random start seq
    #     distance_thres = vis_cfg.get('distance_thres', None) # for candidates
    #     interrupt_thres = vis_cfg.get('interrupt_thres', 10) # for potential interrupts

    #     if random_start:
    #         seq_start_from = np.random.randint(0, seq_length - predict_size + 1)
    #     else:
    #         seq_start_from = min(max(seq_start_from, 0), seq_length - predict_size)

    #     track_map = {}
        
    #     ego_pose = [] # global
    #     for i in range(seq_start_from, seq_length):
    #         ego_data_samples = example_seq[i][0]['data_samples'][0]
    #         infra_data_samples = example_seq[i][1]['data_samples'][0]

    #         ego_bev_path = ego_data_samples.metainfo['bev_path']
    #         infra_bev_path = infra_data_samples.metainfo['bev_path']
    #         if i == seq_start_from:
    #             ego_bev_insmap = np.load(ego_bev_path)['data'] # npz
    #             H, W, _ = ego_bev_insmap.shape # type: ignore
    #             h, w, _ = tuple(grid_size)
    #             minx = (W - w) // 2 - 1
    #             miny = (H - h) // 2 - 1
    #             maxx = w + (W - w) // 2
    #             maxy = h + (H - h) // 2
    #             ego_bev_insmap = ego_bev_insmap[miny:maxy,minx:maxx] # type: ignore
    #             infra_bev_insmap = np.load(infra_bev_path)['data'] # npz
    #             H, W, _ = infra_bev_insmap.shape # type: ignore
    #             h, w, _ = tuple(grid_size)
    #             minx = (W - w) // 2 - 1
    #             miny = (H - h) // 2 - 1
    #             maxx = w + (W - w) // 2
    #             maxy = h + (H - h) // 2
    #             infra_bev_insmap = infra_bev_insmap[miny:maxy,minx:maxx] # type: ignore
            
    #         ego_frame_metainfo = ego_data_samples.metainfo
    #         ego2global = torch.tensor(ego_frame_metainfo['ego2global'], dtype=torch.float32, device=get_device())

    #         infra_frame_metainfo = infra_data_samples.metainfo
    #         infra2global = torch.tensor(infra_frame_metainfo['ego2global'], dtype=torch.float32, device=get_device())
    #         lidar2infra = torch.tensor(infra_frame_metainfo['lidar2ego'], dtype=torch.float32, device=get_device())
    #         lidar2global = infra2global @ lidar2infra # 4 4

    #         if i == seq_start_from:
    #             base_pose = infra2global # fake world，基于第一帧检测起点作为EGO当前帧的所有目标的基准世界位置
    #             # temp = np.eye(3)
    #             # ego2infra = calc_relative_pose(base_pose, ego2global)
    #             # temp[:2, :2] = ego2infra[:2, :2]
    #             # ego_bev_insmap_wrapped = cv2.warpAffine(ego_bev_insmap, temp, (ego_bev_insmap.shape[1], ego_bev_insmap.shape[0]))
    #             # translate = ego2infra[:2, 2] FIXME

    #         lidar2global_rotation_T = lidar2global[:3, :3].T
    #         lidar2global_translation = lidar2global[:3, 3]
    #         infra_data_samples.gt_instances_3d.bboxes_3d.rotate(lidar2global_rotation_T, None)
    #         infra_data_samples.gt_instances_3d.bboxes_3d.translate(lidar2global_translation)

    #         # gt_boxes_center_lidar = data_samples.gt_instances_3d.bboxes_3d.gravity_center.unsqueeze(0) # 1 N 3
    #         # gt_boxes_center_global = simple_points_project(gt_boxes_center_lidar, lidar2global)# 1 N 3

    #         instances = infra_data_samples.gt_instances_3d
    #         if i == seq_start_from: # 第一帧雷达可视范围内检测到了， 之后用的真值认为是预测结果
    #             instances = instances[instances.bbox_3d_isvalid == True] # type: ignore
    #         track_id = instances.track_id
    #         for id in track_id:
    #             if i == seq_start_from: # 启动第一帧跟踪所有可视的对象的id
    #                 track_map[id.item()] = {
    #                     'cmap': cmaps[np.random.randint(0, len(cmaps))],
    #                     'instance': instances[track_id == id.item()],
    #                 }
    #             elif id.item() in track_map:
    #                 if len(track_map[id.item()]['instance']) >= predict_size: # 对于跟踪大于预测之后停止跟踪，认为是当前的motion预测的值
    #                     continue
    #                 track_map[id.item()]['instance'] = InstanceData.cat([track_map[id.item()]['instance'], instances[track_id == id.item()]])

    #         ego_pose.append(ego2global) # type: ignore
        
    #     scatter_trans = torch.tensor(
    #         [[0, 1],
    #          [-1, 0]],
    #          dtype=torch.float32,
    #          device=get_device()
    #     )
    #     ego_pose_rela = calc_relative_pose(base_pose, ego_pose) # type: ignore
    #     ego_pose_rela_center = torch.stack([pose[:2, 3] for pose in ego_pose_rela]) # [7, 2] EGO 规划真值位置序列
    #     ego_start_center = ego_pose_rela_center[0]
    #     base_center = torch.zeros_like(ego_start_center)

    #     ego_pose_rela_center_vis = ego_pose_rela_center @ scatter_trans.T # EGO 规划平面位置序列
    #     ego_pose_rela_x_vis = torch.round(ego_pose_rela_center_vis[:, 0] / voxel_size[0] + grid_size[1] * 0.5).int().cpu().numpy()
    #     ego_pose_rela_y_vis = torch.round(ego_pose_rela_center_vis[:, 1] / voxel_size[1] + grid_size[0] * 0.5).int().cpu().numpy()
    #     base_center_x_vis = torch.round(base_center[0] / voxel_size[0] + grid_size[1] * 0.5).int().cpu().numpy()
    #     base_center_y_vis = torch.round(base_center[1] / voxel_size[1] + grid_size[0] * 0.5).int().cpu().numpy()
    #     if distance_thres != None:
    #         distance_radius = distance_thres / voxel_size[0] # FIXME
    #     interrupt_radius = interrupt_thres / voxel_size[0] # FIXME
        
    #     fig, ax = plt.subplots(1, 1)
    #     ax.imshow(infra_bev_insmap) # type: ignore
    #     ego_cmap = get_cmap('Blues')
    #     c = np.linspace(0.0, 1.0, seq_length - seq_start_from)[::-1]
    #     ax.scatter(ego_pose_rela_x_vis, ego_pose_rela_y_vis, c=c, cmap='Blues', s=6)
    #     circles = [Circle((ego_pose_rela_x_vis[i], ego_pose_rela_y_vis[i]), interrupt_radius, edgecolor=ego_cmap(c[i]), fill=False, linewidth=0.25) for i in range(seq_length - seq_start_from)]
    #     if distance_thres != None:
    #         circle = Circle((base_center_x_vis, base_center_y_vis), distance_radius, edgecolor='cyan', fill=False, linewidth=0.5) # 第一帧的匹配范围绘制 # type: ignore
    #         circles.append(circle)
    #     for circle in circles:
    #         ax.add_patch(circle)

    #     for i, seq in enumerate(list(range(seq_start_from, seq_length))): # 绘制从seq_start_from开始到结束的EGO驾驶规划意图
    #         ax.text(ego_pose_rela_x_vis[i], ego_pose_rela_y_vis[i], s=str(seq), fontsize=3, ha='center', va='center')
    #         ax.text(ego_pose_rela_x_vis[i], ego_pose_rela_y_vis[i], s="  EGO", fontsize=3, ha='left', va='center')

    #     interrupt_info = {}
    #     interrupt_info['timestamps'] = seq_timestamps[seq_start_from : seq_length]
    #     interrupt_info['distance_radius'] = distance_thres
    #     interrupt_info['inside'] = 0
    #     interrupt_info['interrupt_radius'] = interrupt_thres
    #     interrupt_info['predict_size'] = predict_size
    #     interrupt_info['result'] = []

    #     # 开始筛选目标，判断潜在的碰撞风险
    #     for track_id, v in track_map.items():
    #         cmap = v['cmap']
    #         instance = v['instance']
    #         last = len(instance)
    #         if last < predict_size: # 预测长度不足predict_size的跟踪轨迹直接忽略（实际motion输出一定是定长的）
    #             continue
    #         label = instance.labels_3d[0].item()
            
    #         gt_boxes_xyz_global = instance.bboxes_3d.gravity_center.unsqueeze(0) # [1, N, 3] 基于真实世界的位置需要转换到相对于基准位置
    #         gt_boxes_xyz_rela = simple_points_project(gt_boxes_xyz_global, torch.linalg.inv(base_pose)).squeeze(0) # [N, 3] moiton 预测真值位置序列 # type: ignore
    #         gt_boxes_center_rela = gt_boxes_xyz_rela[:, :2]
    #         start_center = gt_boxes_center_rela[0]
    #         if distance_thres != None:
    #             if torch.norm(start_center - base_center) > distance_thres: # 第一帧检测范围内的目标作为潜在候选
    #                 continue
    #         if torch.norm(start_center - ego_start_center) < 0.5: # 排除路侧检测到EGO轨迹和EGO轨迹重合
    #             continue
    #         interrupt_info['inside'] += 1
    #         valid = False
    #         interrupt_seq = []
    #         interrupt_i = []
    #         interrupt_distances = []
    #         for i, seq in enumerate(list(range(seq_start_from, seq_start_from + predict_size))):
    #             center = gt_boxes_center_rela[i] # 这里是跟踪的固定长度
    #             ego_center = ego_pose_rela_center[i] # 这里必须从整个轨迹序列中开始取得
    #             temp_dis = torch.norm(center - ego_center)
    #             if temp_dis <= interrupt_thres:
    #                 valid = True
    #                 interrupt_seq.append(seq)
    #                 interrupt_i.append(i)
    #                 interrupt_distances.append(temp_dis.item())

    #         gt_boxes_center_vis = gt_boxes_center_rela @ scatter_trans.T # [N, 2] moiton 预测平面位置序列
    #         gt_boxes_x_vis = torch.round(gt_boxes_center_vis[:, 0] / voxel_size[0] + grid_size[1] * 0.5).int().cpu().numpy()
    #         gt_boxes_y_vis = torch.round(gt_boxes_center_vis[:, 1] / voxel_size[1] + grid_size[0] * 0.5).int().cpu().numpy()
            
    #         if not valid:
    #             ax.scatter(gt_boxes_x_vis, gt_boxes_y_vis, c='black', s=1)
    #             for i, seq in enumerate(list(range(seq_start_from, seq_start_from + predict_size))):
    #                 ax.text(gt_boxes_x_vis[i], gt_boxes_y_vis[i], s="  " + str(seq), fontsize=2.0, ha='left', va='top')
    #                 ax.text(gt_boxes_x_vis[i], gt_boxes_y_vis[i], s="  " + str(track_id), fontsize=2.0, ha='left', va='bottom')
    #         else:
    #             interrupt_info['result'].append(dict(
    #                 id = track_id,
    #                 label = label,
    #                 interrupt_timestamps = interrupt_seq,
    #                 interrupt_distances = interrupt_distances
    #             ))
    #             ax.scatter(gt_boxes_x_vis, gt_boxes_y_vis, c=np.arange(predict_size)[::-1], cmap=cmap, s=2)
    #             for i, seq in enumerate(list(range(seq_start_from, seq_start_from + predict_size))):
    #                 ax.text(gt_boxes_x_vis[i], gt_boxes_y_vis[i], s="  " + str(seq), fontsize=2.0, ha='left', va='top')
    #                 ax.text(gt_boxes_x_vis[i], gt_boxes_y_vis[i], s="  " + str(track_id), fontsize=2.0, ha='left', va='bottom')
    #                 if i in interrupt_i:
    #                     ax.scatter(gt_boxes_x_vis[i], gt_boxes_y_vis[i], marker='o', facecolors='none', edgecolors='blue', s=20, linewidths= 0.5) # type:ignore
    #     from pprint import pprint
    #     pprint(interrupt_info)
        
    #     fig.savefig('./motion.png', dpi = 800)
        
    #     import pdb
    #     pdb.set_trace()

        
        