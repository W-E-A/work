# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch.nn as nn
from typing import Optional, Union, Dict, Tuple
from .sub_modules import PillarVFE
from .sub_modules import PointPillarScatter
from .sub_modules import ResNetBEVBackbone
from .sub_modules import DownsampleConv
from .fuse_modules import Where2comm
import torch
import numpy as np
from mmengine.model import BaseModel, BaseModule, ModuleList
from mmdet3d.registry import MODELS
from mmengine.optim import OptimWrapper
from projects.Where2comm.models.point_pillar_loss import PointPillarLoss
from projects.Where2comm.models.utils import postprocess, temp_vis, warp_affine

IDX = 0

VFE_CFG = {
    'voxel_size': [0.4, 0.4, 4],
    'lidar_range': [-100.8, -40, -3, 100.8, 40, 1],
    'anchor_number': 2,
    'downsample_rate' : 2,
    # 'max_cav': 2,
    # 'compression': 0,
    # 'vehicle_fix': False,
    # 'infra_fix': False,
    # 'fusion_fix': False,
    'use_norm': True,
    'with_distance': False,
    'use_absolute_xyz': True,
    'num_filters': [64],
}

GRID_SIZE = (np.array(VFE_CFG['lidar_range'][3:6]) - np.array(VFE_CFG['lidar_range'][0:3])) / np.array(VFE_CFG['voxel_size'])
GRID_SIZE = np.round(GRID_SIZE).astype(np.int64)

SCATTER_CFG = {
    'num_features': 64,
    'grid_size': GRID_SIZE,
}

BACKBONE_CFG = {
    'resnet': True,
    'layer_nums': [3, 4, 5],
    'layer_strides': [2, 2, 2],
    'num_filters': [64, 128, 256],
    'upsample_strides': [1, 2, 4],
    'num_upsample_filter': [128, 128, 128],
    'compression': 0,
    'voxel_size': VFE_CFG['voxel_size'],
}

SHRINK_CFG = {
    'kernal_size': [ 3 ],
    'stride': [ 1 ],
    'padding': [ 1 ],
    'dim': [ 256 ],
    'input_dim': 384, # 128 * 3
}

TEST_CFG = {
    'agent_threshold' : [0.01, 0.01],
    'fusion_threshold' : 0.1,
    'order' : 'hwl',
    'nms_threshold' : 0.15,
    'lidar_range' : VFE_CFG['lidar_range'],
    'only_vis' : True
}

class PointPillar(nn.Module):
    def __init__(self):
        super(PointPillar, self).__init__()
        # PIllar VFE
        self.pillar_vfe = PillarVFE(VFE_CFG,
                                    num_point_features=4,
                                    voxel_size=VFE_CFG['voxel_size'],
                                    point_cloud_range=VFE_CFG['lidar_range'])
        self.scatter = PointPillarScatter(SCATTER_CFG)
        self.backbone = ResNetBEVBackbone(BACKBONE_CFG, 64)
        # used to downsample the feature map for efficient computation
        self.shrink_conv = DownsampleConv(SHRINK_CFG)
        self.cls_head = nn.Conv2d(128 * 2, VFE_CFG['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * VFE_CFG['anchor_number'],
                                  kernel_size=1)
 
    def forward(self, batch_dict):
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict) # -> ['pillar_features'] [M, 64]
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict) # -> ['spatial_features'] [N, C, H, W] C=64
        batch_dict = self.backbone(batch_dict) # -> ['spatial_features_2d'] [N, C, H, W]
        # N, C, H', W'. [N, 384, 100, 252] # H,W shrink after spatial downsample
        spatial_features = batch_dict['spatial_features']
        spatial_features_2d = batch_dict['spatial_features_2d']
        
        # downsample feature to reduce memory
        spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # spatial_features_2d is [sum(cav_num), 256, 50, 176]
        # output only contains ego
        # [B, 256, 50, 176]
        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)
        return psm, rm, spatial_features, spatial_features_2d

@MODELS.register_module()
class MyModel(BaseModule):
    def __init__(self,
                 pts_voxel_encoder,
                 pts_middle_encoder,
                 pts_backbone,
                 pts_shrink_module,
                 pts_detect_module
                 ):
        super(MyModel, self).__init__()
        self.voxel_encoder = MODELS.build(pts_voxel_encoder)
        self.middle_encoder = MODELS.build(pts_middle_encoder)
        self.backbone = MODELS.build(pts_backbone)
        self.shrink_module = MODELS.build(pts_shrink_module)
        self.detect_module = MODELS.build(pts_detect_module)
    
    def forward(self, batch_dict):
        pillar_feat = self.voxel_encoder(
            batch_dict['voxel_features'], # [pls, M, C]
            batch_dict['voxel_num_points'], # [pls, ]
            batch_dict['voxel_coords'], # [pls, 1 + 3] b z y x
        )
        scatter_feat = self.middle_encoder(
            pillar_feat,
            batch_dict['voxel_coords'],
            batch_dict['batch_size']
        )
        bev_feat = self.backbone(
            scatter_feat
        )
        bev_feat = self.shrink_module(
            bev_feat
        )
        psm, rm = self.detect_module(
            bev_feat
        )
        return psm, rm, scatter_feat, bev_feat
        
@MODELS.register_module()
class PointPillarWhere2comm(BaseModel):
    def __init__(self, my_model = None, pts_fusion_module = None, pts_comm_module = None, pts_detect_module = None):
        super(PointPillarWhere2comm, self).__init__()
        # used to downsample the feature map for efficient computation

        self.model_infra = MODELS.build(my_model)
        self.model_vehicle = MODELS.build(my_model)
        # self.model_infra = PointPillar()
        # self.model_vehicle = PointPillar()
        # self.fusion_net = TransformerFusion(args['fusion_args'])

        # self.fusion_net = Where2comm()

        # self.cls_head = nn.Conv2d(128 * 2, VFE_CFG['anchor_number'], kernel_size=1)
        # self.reg_head = nn.Conv2d(128 * 2, 7 * VFE_CFG['anchor_number'], kernel_size=1)

        self.comm = True
        self.comm_module = MODELS.build(pts_comm_module)
        self.fusion_detect_module = MODELS.build(pts_detect_module)
        self.fusion_list = ModuleList([MODELS.build(pts_fusion_module) for i in range(3)])

        self.loss_module = PointPillarLoss()
    
    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self(data, mode='loss')
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        return log_vars
    
    def val_step(self, data: Union[tuple, dict, list]) -> list:
        data = self.data_preprocessor(data, False)
        return self(data, mode='predict')
    
    def test_step(self, data: Union[dict, tuple, list]) -> list:
        data = self.data_preprocessor(data, False)
        return self(data, mode='predict')
    
    def parse_losses(
        self, *args, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        loss, log_vars = super().parse_losses(*args, **kwargs)
        log_vars.pop('loss')

        return loss, log_vars

    def forward(self,
                inputs: dict,
                mode: str = 'tensor'):
        
        """
        Calculate losses from a batch of inputs and data samples.

        dict_keys([
            'agents', # List[Tuple]
            'ego_ids', # List[int]
            'pose_list', # tensor(B, A, 4, 4)
            'anchor_box', # tensor(B, H, W, 2, 7)
            'gt_boxes', # tensor(B, max, 7)
            'gt_mask', # tensor(B, max)
            'gt_object_ids', # List[int]
            'pc_list', # 
            'pos_equal_one', # tensor(B, H, W, 2)
            'labels', # tensor(B, H, W, 14)
            '0_gt_boxes'
            '0_gt_mask',
            '0_gt_object_ids',
            '0_pos_equal_one',
            '0_labels',
            '1_gt_boxes',
            '1_gt_mask',
            '1_gt_object_ids',
            '1_pos_equal_one',
            '1_labels',
            'vis_pc_list',
            'voxel_features', # tensor(B, N, C)
            'voxel_coords', # tensor(B, N, 1 + 3)
            'voxel_num_points', # tensor(B, N, )
            'lidar_to_world_matrix']) # tensor(B, 4, 4)
        """

        batch_dict = inputs

        batch_ego_ids = batch_dict['ego_ids']
        batch_size = len(batch_ego_ids)

        batch_agent_rela_pose = []
        for b in range(batch_size):
            agent_rela_pose = []
            for idx in range(2):
                agent_rela_pose.append(torch.linalg.inv(batch_dict['pose_list'][b][idx])) # [4, 4]
            batch_agent_rela_pose.append(torch.stack(agent_rela_pose, dim=0)) # [agents, 4, 4]
        batch_agent_rela_pose = torch.stack(batch_agent_rela_pose, dim=0) # [bs, agents, 4, 4]

        record_len = torch.tensor([2 for b in range(batch_size)], device=batch_agent_rela_pose.device)
        pairwise_t_matrix = torch.tile(torch.eye(4, dtype=torch.float32), (batch_size, 2, 2, 1, 1))
        pairwise_t_matrix[:, 0, ...] = batch_agent_rela_pose

        batch_dict_v = {
            'voxel_features' : batch_dict['voxel_features'][0],
            'voxel_num_points' : batch_dict['voxel_num_points'][0],
            'voxel_coords' : batch_dict['voxel_coords'][0],
            'record_len' : record_len,
            'batch_size' : batch_size
        }

        batch_dict_i = {
            'voxel_features' : batch_dict['voxel_features'][1],
            'voxel_num_points' : batch_dict['voxel_num_points'][1],
            'voxel_coords' : batch_dict['voxel_coords'][1],
            'record_len' : record_len,
            'batch_size' : batch_size
        }

        # [batch*H*W, points, channel] [batch*H*W, id+z+y+x] [batch*H*W,]
        psm_single_v, rm_single_v, spatial_features_v, spatial_features_2d_v = self.model_vehicle(batch_dict_v)
        # [batch*H*W, points, channel] [batch*H*W, id+z+y+x] [batch*H*W,]
        psm_single_i, rm_single_i, spatial_features_i, spatial_features_2d_i= self.model_infra(batch_dict_i)

        # psm: 空间每个位置的2个anchor的置信度
        # rm: 空间中每个位置anchor的回归值xyz hwl yaw
        # spatial_features: 未经过主干编码的空间特征
        # spatial_features_2d: 经过主干编码的特征

        psm_single, rm_single, spatial_features, spatial_features_2d = [], [], [], []
        for i in range(psm_single_v.shape[0]):
            # psm_single.append(psm_single_v[i, :, :, :])
            # psm_single.append(psm_single_i[i, :, :, :])
            psm_single.append(torch.stack([psm_single_v[i, :, :, :], psm_single_i[i, :, :, :]]))
        for i in range(spatial_features_v.shape[0]):
            # spatial_features.append(spatial_features_v[i, :, :, :])
            # spatial_features.append(spatial_features_i[i, :, :, :])
            spatial_features.append(torch.stack([spatial_features_v[i, :, :, :], spatial_features_i[i, :, :, :]]))
        for i in range(spatial_features_2d_v.shape[0]):
            # spatial_features_2d.append(spatial_features_2d_v[i, :, :, :])
            # spatial_features_2d.append(spatial_features_2d_i[i, :, :, :])
            spatial_features_2d.append(torch.stack([spatial_features_2d_v[i, :, :, :], spatial_features_2d_i[i, :, :, :]]))
        for i in range(rm_single_v.shape[0]):
            rm_single.append(torch.stack([rm_single_v[i, :, :, :], rm_single_i[i, :, :, :]]))
        psm_single = torch.stack(psm_single)
        spatial_features = torch.stack(spatial_features)
        spatial_features_2d = torch.stack(spatial_features_2d)
        rm_single = torch.stack(rm_single)

        # torch.Size([8, 2, 100, 252])
        # (Pdb) print(spatial_features.shape)
        # torch.Size([8, 64, 200, 504])
        # (Pdb) print(spatial_features_2d.shape)
        # torch.Size([8, 256, 100, 252])

        # import pdb
        # pdb.set_trace()


        
        # fused_feature, communication_rates = self.fusion_net(spatial_features,
        #                                     psm_single,
        #                                     record_len,
        #                                     pairwise_t_matrix, 
        #                                     self.model_vehicle.backbone,
        #                                     None)


        if self.comm:
            communication_masks, ori_communication_maps, communication_maps, communication_rates = \
            self.comm_module(psm_single, 0)
            # [bs, agents, 1, H, W] in {0, 1}
            # [bs, agents, 1, H, W] in [0, 1]
            # [bs, agents, 1, H, W] in [0, 1] larger
            # [bs, agents, ]

        batch_agent_rela_pose = []
        for b in range(batch_size):
            agent_rela_pose = []
            for idx in range(2):
                agent_rela_pose.append(torch.linalg.inv(batch_dict['pose_list'][b][idx])) # [4, 4]
            batch_agent_rela_pose.append(torch.stack(agent_rela_pose, dim=0)) # [agents, 4, 4]
        batch_agent_rela_pose = torch.stack(batch_agent_rela_pose, dim=0) # [bs, agents, 4, 4]

        encode_feats = []
        for b in range(batch_size):
            encode_feats.append(self.model_vehicle.backbone.resnet(spatial_features[b]))

        ups = []
        for i in range(3):
            layer_fuse = []
            for b in range(batch_size):
                rela_pose = batch_agent_rela_pose[b] # [agents, 4, 4]
                agent_feats = encode_feats[b][i]
                if self.comm and i == 0:
                    masked_agent_feats = agent_feats * communication_masks[b] # type: ignore
                else:
                    masked_agent_feats = agent_feats
                
                masked_agent_feats_wrap = warp_affine(masked_agent_feats, # [agents, C', H, W], [agents, C'*2, H/2, W/2], [agents, C'*4, H/4, W/4]
                                    rela_pose, # [agents, 4, 4]
                                    VFE_CFG['downsample_rate'],
                                    VFE_CFG['voxel_size'])
                
                layer_fuse.append(self.fusion_list[i](masked_agent_feats_wrap, 0))
                # [1, C', H, W], [1, C'*2, H/2, W/2], [1, C'*4, H/4, W/4]
            layer_fuse = torch.stack(layer_fuse, dim=0) # [bs, C', H, W], [bs, C'*2, H/2, W/2], [bs, C'*4, H/4, W/4]
            if len(self.model_vehicle.backbone.deblocks) > 0: # type: ignore
                ups.append(self.model_vehicle.backbone.deblocks[i](layer_fuse)) # type: ignore
            else:
                ups.append(layer_fuse)
                
        if len(ups) > 1:
            fused_feature = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            fused_feature = ups[0]
        
        if len(self.model_vehicle.backbone.deblocks) > 3: # type: ignore
            fused_feature = self.model_vehicle.backbone.deblocks[-1](fused_feature) # type: ignore
        # decode_feat = self.backbone_list[ego_idx](x_fuse, False, True) # [bs, C'*2*num_layer, H, W]

        # downsample feature to reduce memory
        fused_feature = self.model_vehicle.shrink_module(fused_feature)
        # fused_feature = self.model_vehicle.shrink_conv(fused_feature)
        
        # psm = self.cls_head(fused_feature)
        # rm = self.reg_head(fused_feature)
        psm, rm = self.fusion_detect_module(fused_feature)

        if mode == 'loss':
            loss_dict = {}

            fusion_result = {
                'psm': psm,
                'rm': rm,
                'pos_equal_one': batch_dict['pos_equal_one'],
                'labels': batch_dict['labels'],
            }
            _, fusion_cls_loss, fusion_reg_loss = self.loss_module(fusion_result)

            loss_dict['Fuse_Closs'] = fusion_cls_loss
            loss_dict['Fuse_Rloss'] = fusion_reg_loss

            v_result = {
                'psm': psm_single_v,
                'rm': rm_single_v,
                'pos_equal_one': batch_dict['0_pos_equal_one'],
                'labels': batch_dict['0_labels'],
            }
            _, v_cls_loss, v_reg_loss = self.loss_module(v_result)

            loss_dict['V_Closs'] = v_cls_loss
            loss_dict['V_Rloss'] = v_reg_loss

            i_result = {
                'psm': psm_single_i,
                'rm': rm_single_i,
                'pos_equal_one': batch_dict['1_pos_equal_one'],
                'labels': batch_dict['1_labels'],
            }
            _, i_cls_loss, i_reg_loss = self.loss_module(i_result)

            loss_dict['I_Closs'] = i_cls_loss
            loss_dict['I_Rloss'] = i_reg_loss

            # if self.comm:
            #     for idx, id in enumerate(self.co_mask):
            #         loss_dict[f"{agents[id][:3]}_CR"] = communication_rates[idx] # type: ignore

            return loss_dict
        elif mode == 'predict':

            assert batch_size == 1, "Only supports a single test/val at a time"
            
            corner_pred_3d_final, masked_cls_pred_final, masked_gt_corner_final = \
            postprocess(psm, rm, TEST_CFG['fusion_threshold'], -1, TEST_CFG, batch_dict) # type: ignore

            v_corner_pred_3d_final, v_masked_cls_pred_final, v_masked_gt_corner_final = \
            postprocess(psm_single_v, rm_single_v, TEST_CFG['agent_threshold'][0], 0, TEST_CFG, batch_dict) # type: ignore

            i_corner_pred_3d_final, i_masked_cls_pred_final, i_masked_gt_corner_final = \
            postprocess(psm_single_i, rm_single_i, TEST_CFG['agent_threshold'][1], 1, TEST_CFG, batch_dict) # type: ignore

            global IDX
            START = 13
            END = 23
            vis_dict = {
                'pc' : batch_dict['vis_pc_list'][0][0],
                'pc_range' : TEST_CFG['lidar_range'],
                'pred_corner' : corner_pred_3d_final,
                'gt_corner' : masked_gt_corner_final,
                'psm' : psm,
                'feat' : fused_feature
            }
            valid = temp_vis(
                IDX,
                -1,
                vis_dict,
                False,
                save_start = START,
                save_end = END,
                save_wrap = False
            )
            
            # for idx, id in enumerate(self.co_mask):
            #     vis_dict = {
            #         'pc' : batch_dict['vis_pc_list'][0][id],
            #         'pc_range' : TEST_CFG['lidar_range'],
            #         'pred_corner' : corner_pred_3d_final_list[idx],
            #         'gt_corner' : masked_gt_corner_final_list[idx],
            #         'cmap' : communication_maps[:, idx, ...], # type: ignore
            #         'mask' : communication_masks[:, idx, ...], # type: ignore
            #         'psm' : psms[:, idx, ...],
            #         'feat' : bev_feats[:, idx, ...],
            #         # 'feat' : masked_agent_feats[idx].unsqueeze(0),
            #         # 'feat' : masked_agent_feats_wrap[idx].unsqueeze(0),
            #         'rela_pose' : batch_agent_rela_pose[:, idx, ...], # type: ignore
            #         'downsample_rate' : self.downsample_rate,
            #         'voxel_size' : self.voxel_size
            #     }
            #     valid = temp_vis(
            #         IDX,
            #         id,
            #         vis_dict,
            #         True,
            #         save_start = START,
            #         save_end = END,
            #         save_wrap = True if id != ego_id else False
            #     )

            if not valid and TEST_CFG['only_vis'] == True:
                import pdb
                pdb.set_trace()
            IDX += 1

            if corner_pred_3d_final.shape[0] == 0:
                corner_pred_3d_final = None
                masked_cls_pred_final = None

            return [corner_pred_3d_final, masked_cls_pred_final, masked_gt_corner_final]
            
        elif mode == 'tensor':
            return None
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')