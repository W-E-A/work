
from typing import Optional, Union, Dict, Tuple
from mmdet3d.registry import MODELS
from mmengine.model import BaseModel, ModuleList
import torch
import torch.nn as nn
from .utils import warp_affine, decode_reg_result, boxes_to_corners_baseline, \
corner_to_standup_box, remove_large_pred_bbx, remove_bbx_abnormal_z, nms_rotated, \
get_mask_for_boxes_within_range
from mmengine.optim import OptimWrapper
# from mmengine.visualization import Visualizer
from projects.Where2comm.visualization import visualize
import time
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import os

IDX = 0

@MODELS.register_module()
class Where2comm(BaseModel):
    def __init__(self,
                 co_agents: Union[int, list],
                 multi_scale: bool,
                 voxel_size: list,
                 downsample_rate: int,
                 pts_voxel_encoder: dict,
                 pts_middle_encoder: dict,
                 pts_backbone: dict,
                 pts_fusion_module: dict,
                 pts_detect_module: dict,
                 pts_loss_module: dict,
                 pts_comm_module: Optional[dict] = None,
                 pts_shrink_module: Optional[dict] = None,
                 pts_compress_module: Optional[dict] = None,
                 pts_dcn_module: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):
        super(Where2comm, self).__init__(data_preprocessor, init_cfg)
        
        self.co_mask = []
        if isinstance(co_agents, int):
            self.co_mask += list(range(co_agents))
        elif isinstance(co_agents, list):
            assert all(id >= 0 for id in co_agents), f'Agent id must >= 0, got "{co_agents}"'
            self.co_mask += co_agents
        else:
            raise TypeError(f"Invalid co_agnets type")
        
        self.co_length = len(self.co_mask)
        assert self.co_length > 1,"Co length must >1 ."

        # self.multi_scale = multi_scale
        self.voxel_size = voxel_size
        self.downsample_rate = downsample_rate

        self.voxel_encoder_list = ModuleList([MODELS.build(pts_voxel_encoder) for i in range(self.co_length)])
        self.middle_encoder_list = ModuleList([MODELS.build(pts_middle_encoder) for i in range(self.co_length)])
        self.backbone_list = ModuleList([MODELS.build(pts_backbone) for i in range(self.co_length)])
        self.detect_list = ModuleList([MODELS.build(pts_detect_module) for i in range(self.co_length)])

        self.shrink = False
        self.shrink_list = []
        if pts_shrink_module:
            self.shrink = True
            self.shrink_list = ModuleList([MODELS.build(pts_shrink_module) for i in range(self.co_length)])
        
        self.compress = False
        self.compress_list = []
        if pts_compress_module:
            self.compress = True
            self.compress_list = ModuleList([MODELS.build(pts_compress_module) for i in range(self.co_length)])

        self.comm = False
        if pts_comm_module:
            self.comm = True
            self.pts_comm_module = MODELS.build(pts_comm_module)
        self.pts_fusion_module = MODELS.build(pts_fusion_module)
        self.fusion_detect_module = MODELS.build(pts_detect_module)
        self.loss_module = MODELS.build(pts_loss_module)

        self.test_cfg = test_cfg

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
            'pc', # 
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
            'vis_pc',
            'voxel_features', # tensor(B, N, C)
            'voxel_coords', # tensor(B, N, 1 + 3)
            'voxel_num_points', # tensor(B, N, )
            'lidar_to_world_matrix']) # tensor(B, 4, 4)
        """

        batch_dict = inputs

        batch_agents = batch_dict['agents']
        batch_ego_ids = batch_dict['ego_ids']
        ego_id = batch_ego_ids[0]
        agents = batch_dict['agents'][0]
        agent_length = len(batch_agents[0])
        batch_size = len(batch_ego_ids)

        assert all(id in self.co_mask for id in batch_ego_ids)
        assert len(self.co_mask) <= agent_length
        assert ego_id in self.co_mask
        ego_idx = self.co_mask.index(ego_id)

        agent_result = []

        for idx, id in enumerate(self.co_mask):
            result = {}
            pillar_feat = self.voxel_encoder_list[idx](
                batch_dict['voxel_features'][id], # [pls, M, C]
                batch_dict['voxel_num_points'][id], # [pls, ]
                batch_dict['voxel_coords'][id], # [pls, 1 + 3] b z y x
            ) # [pls, C']
            scatter_feat = self.middle_encoder_list[idx](
                pillar_feat, # [pls, C']
                batch_dict['voxel_coords'][id], # [pls, 1 + 3] b z y x
                batch_size # bs
            ) # [bs, C', H, W]
            bev_feat = self.backbone_list[idx](
                scatter_feat
            ) # [bs, C'', H, W]
            if self.shrink:
                bev_feat = self.shrink_list[idx](
                    bev_feat
                )
            if self.compress:
                bev_feat = self.compress_list[idx](
                    bev_feat
                )
            psm, rm = self.detect_list[idx](
                bev_feat
            )
            result['scatter_feat'] = scatter_feat # [bs, C', H, W]
            result['bev_feat'] = bev_feat # [bs, C'', H, W]
            result['psm'] = psm # [bs, 2, H, W]
            result['rm'] = rm # [bs, 7*2, H, W]
            agent_result.append(result)

        agent_psm = [result['psm'] for result in agent_result] # [bs, 2, H, W]
        if self.comm:
            communication_masks, communication_maps, communication_rates = self.pts_comm_module(agent_psm, ego_idx)

        x_fuse = []
        for b in range(batch_size):

            agent_rela_pose = []
            agent_feats = []
            if self.comm:
                agent_mask = []
            for idx, id in enumerate(self.co_mask):
                agent_rela_pose.append(torch.linalg.inv(batch_dict['pose_list'][b][id])) # [4, 4]
                agent_feats.append(agent_result[idx]['bev_feat'][b]) # [C'', H, W]
                if self.comm:
                    agent_mask.append(communication_masks[idx][b]) # [1, H, W] # type: ignore
            agent_rela_pose = torch.stack(agent_rela_pose, dim=0) # [agents, 4, 4]
            agent_feats = torch.stack(agent_feats, dim=0) # [agents, C'', H, W]
            if self.comm:
                agent_mask = torch.stack(agent_mask, dim=0) # [agents, 1, H, W] # type: ignore
                agent_feats = agent_feats * agent_mask # [agents, C'', H, W]

            agent_feats = warp_affine(agent_feats, # [agents, C'', H, W]
                                    agent_rela_pose, # [agents, 4, 4]
                                    self.downsample_rate,
                                    self.voxel_size)
            
            if mode == 'predict':

                temp_feats = warp_affine(agent_mask, # [agents, 1, H, W] # type: ignore
                                        agent_rela_pose, # [agents, 4, 4]
                                        self.downsample_rate,
                                        self.voxel_size)
            
            x_fuse.append(self.pts_fusion_module(agent_feats, ego_idx))# [agents, C'', H, W]
            # [C'', H, W]

        # TODO choose fusion method

        x_fuse = torch.stack(x_fuse, dim=0) # [bs, C'', H, W]
        fusion_psm, fusion_rm = self.fusion_detect_module(x_fuse) # [bs, 2, H, W] # [bs, 2*7, H, W]

        if mode == 'loss':

            loss_dict = {}

            fusion_result = {
                'psm': fusion_psm,
                'rm': fusion_rm,
                'pos_equal_one': batch_dict['pos_equal_one'],
                'labels': batch_dict['labels'],
            }
            _, fusion_cls_loss, fusion_reg_loss = self.loss_module(fusion_result)

            loss_dict['Fuse_Closs'] = fusion_cls_loss
            loss_dict['Fuse_Rloss'] = fusion_reg_loss

            for idx, id in enumerate(self.co_mask):
                input = {
                    'psm': agent_result[idx]['psm'],
                    'rm': agent_result[idx]['rm'],
                    'pos_equal_one': batch_dict[f'{id}_pos_equal_one'],
                    'labels': batch_dict[f'{id}_labels'],
                }
                _, cls_loss, reg_loss = self.loss_module(input)

                loss_dict[f'{agents[id][:3]}_Closs'] = cls_loss
                loss_dict[f'{agents[id][:3]}_Rloss'] = reg_loss

            if self.comm:
                for idx, id in enumerate(self.co_mask):
                    loss_dict[f"{agents[id][:3]}_CR"] = communication_rates[idx] # type: ignore

            return loss_dict
        
        elif mode == 'predict':

            assert batch_size == 1, "Only supports a single test/val at a time"

            anchor_box = batch_dict['anchor_box'][0] # [H, W, 2, 7] xyzhwly
            
            fusion_cls_pred = torch.sigmoid(fusion_psm.permute(0, 2, 3, 1).contiguous()) # [1, 2, H, W] -> [1, H, W, 2]
            fusion_cls_pred = fusion_cls_pred.reshape(1, -1) # [1, H*W*2]



            fusion_box_pred = decode_reg_result(fusion_rm, anchor_box) # [1, 14, H, W] [H, W, 2, 7] xyzhwly -> []
            
            assert self.test_cfg != None
            assert 'score_threshold' in self.test_cfg # type: ignore
            assert 'order' in self.test_cfg # type: ignore
            assert 'nms_threshold' in self.test_cfg # type: ignore
            assert 'lidar_range' in self.test_cfg # type: ignore
            score_threshold = self.test_cfg['score_threshold'] # type: ignore
            order = self.test_cfg['order'] # type: ignore
            nms_threshold = self.test_cfg['nms_threshold'] # type: ignore
            lidar_range = self.test_cfg['lidar_range'] # type: ignore

            mask = torch.gt(fusion_cls_pred, score_threshold) # [1, H*W*2]
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)  # [1, H*W*2, 7]

            masked_cls_pred = torch.masked_select(fusion_cls_pred[0], mask[0]) # [N, ]
            masked_box_pred = torch.masked_select(fusion_box_pred[0], mask_reg[0]).reshape(-1, 7) # [N, 7]

            if len(masked_box_pred) != 0:

                corner_pred_3d = boxes_to_corners_baseline(masked_box_pred, order) # [N, 8, 3]

                box_pred_2d = corner_to_standup_box(corner_pred_3d) # [N, 4]

                # box_pred_2d_score = torch.cat([box_pred_2d, masked_cls_pred], dim=1) # [N, 5]

                # import pdb
                # pdb.set_trace()

                keep_index_1 = remove_large_pred_bbx(corner_pred_3d)
                # keep_index_2 = remove_bbx_abnormal_z(corner_pred_3d)
                # keep_index = torch.logical_and(keep_index_1, keep_index_2)

                corner_pred_3d_filtered = corner_pred_3d[keep_index_1] # [n, 8, 3]
                masked_cls_pred_filtered = masked_cls_pred[keep_index_1] # [n, ]

                # import pdb
                # pdb.set_trace()

                keep_index = nms_rotated(corner_pred_3d_filtered, masked_cls_pred_filtered, nms_threshold)
                
                # import pdb
                # pdb.set_trace()

                corner_pred_3d_final = corner_pred_3d_filtered[keep_index] # [f, 8, 3]
                masked_cls_pred_final = masked_cls_pred_filtered[keep_index] # [f, ]

                # filter out the prediction out of the range.
                mask = get_mask_for_boxes_within_range(corner_pred_3d_final, lidar_range)
                corner_pred_3d_final = corner_pred_3d_final[mask, :, :]  # [f', 8, 3]
                masked_cls_pred_final = masked_cls_pred_final[mask] # [f', ]

            else:
                corner_pred_3d_final = torch.empty()
                masked_cls_pred_final = torch.empty()

            gt_boxes = batch_dict['gt_boxes'][0] # [100, 7]
            gt_mask = batch_dict['gt_mask'][0] # [100, ]
            gt_object_ids = batch_dict['gt_object_ids'][0] # [N, ]
            masked_gt_boxes = gt_boxes[gt_mask == 1] # [N, 7]

            masked_gt_corner = boxes_to_corners_baseline(masked_gt_boxes, order)

            mask = get_mask_for_boxes_within_range(masked_gt_corner, lidar_range)

            masked_gt_corner_final = masked_gt_corner[mask, :, :]

            # if len(box_pred_2d_list) ==0 or len(box_pred_3d_list) == 0:
            #     return None, None
            
            # # adding dir classifier
            # if 'dm' in output_dict[cav_id].keys() and len(boxes3d) !=0:
            #     dir_offset = self.params['dir_args']['dir_offset']
            #     num_bins = self.params['dir_args']['num_bins']

            #     dm  = output_dict[cav_id]['dm'] # [N, H, W, 4]
            #     dir_cls_preds = dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins) # [1, N*H*W*2, 2]
            #     dir_cls_preds = dir_cls_preds[mask]
            #     # if rot_gt > 0, then the label is 1, then the regression target is [0, 1]
            #     dir_labels = torch.max(dir_cls_preds, dim=-1)[1]  # indices. shape [1, N*H*W*2].  value 0 or 1. If value is 1, then rot_gt > 0
                
            #     period = (2 * np.pi / num_bins) # pi
            #     dir_rot = limit_period(
            #         boxes3d[..., 6] - dir_offset, 0, period
            #     ) # 限制在0到pi之间
            #     boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype) # 转化0.25pi到2.5pi
            #     boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi) # limit to [-pi, pi]

            # process each agent TODO

            # return data batch ??? TODO

            global IDX

            if IDX < 10:
            # if 13 <= IDX and IDX < 23:
                vis_pc = batch_dict['vis_pc_list'][0][ego_idx]

                path = f'./temp_vis/vis_{IDX}'
                os.makedirs(path, exist_ok=True)

                save_path_bev = osp.join(path, 'bev.png')
                save_path_3d = osp.join(path, '3d.png')
                if self.comm:
                    save_path_ego_psm = osp.join(path, 'ego_psm.png')
                    save_path_inf_psm = osp.join(path, 'inf_psm.png')
                    save_path_fuse_psm = osp.join(path, 'fuse_psm.png')
                    save_path_aff_mask = osp.join(path, 'aff_mask.png')
                
                visualize(corner_pred_3d_final, masked_gt_corner_final, vis_pc, lidar_range, save_path_bev, 'bev', vis_gt_box=True, vis_pred_box=True, left_hand=False)
                visualize(corner_pred_3d_final, masked_gt_corner_final, vis_pc, lidar_range, save_path_3d, '3d', vis_gt_box=True, vis_pred_box=True, left_hand=False)
                
                if self.comm:
                    ego_psm = communication_maps[ego_idx][0].permute(1, 2, 0).contiguous().squeeze(-1).cpu().numpy() # type: ignore
                    ego_psm = np.flipud(ego_psm)
                    plt.imsave(save_path_ego_psm, ego_psm) # type: ignore

                    inf_psm = communication_maps[1][0].permute(1, 2, 0).contiguous().squeeze(-1).cpu().numpy() # type: ignore
                    inf_psm = np.flipud(inf_psm)
                    plt.imsave(save_path_inf_psm, inf_psm) # type: ignore

                    fuse_psm = torch.sigmoid(fusion_psm[0].permute(1, 2, 0).contiguous())
                    fuse_psm = torch.max(fuse_psm, dim=-1).values
                    fuse_psm = np.flipud(fuse_psm.cpu().numpy())
                    plt.imsave(save_path_fuse_psm, fuse_psm) # type: ignore

                    # temp = agent_feats[0].permute(1, 2, 0).contiguous().squeeze(-1) # type: ignore
                    # temp = torch.sigmoid(torch.mean(temp,dim=-1)).cpu().numpy()

                    aff_mask = temp_feats[1].permute(1, 2, 0).contiguous().squeeze(-1).cpu().numpy() # type: ignore
                    # aff_mask = temp
                    aff_mask = np.flipud(aff_mask)
                    plt.imsave(save_path_aff_mask, aff_mask) # type: ignore
            else:
            # elif IDX >= 23:
                import pdb
                pdb.set_trace()
            
            IDX += 1

            return corner_pred_3d_final, masked_cls_pred_final, masked_gt_corner_final
        
        elif mode == 'tensor':

            tensor_list = []
            tensor_list.append(fusion_psm)
            tensor_list.append(fusion_rm)
            for idx, id in enumerate(self.co_mask):
                tensor_list.append(agent_result[idx]['psm'])
                tensor_list.append(agent_result[idx]['rm'])

            return tuple(tensor_list)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')