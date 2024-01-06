
from typing import Optional, Union, Dict, Tuple
from mmdet3d.registry import MODELS
from mmengine.registry import MODEL_WRAPPERS
from mmengine.model import BaseModel, ModuleList
from torch.nn.parallel import DistributedDataParallel
import torch
import torch.nn as nn
from .utils import warp_affine, postprocess, temp_vis
from mmengine.optim import OptimWrapper
from mmengine.logging import print_log
import logging

IDX = 0

@MODELS.register_module()
class Where2comm(BaseModel):
    check_flag = False
    def __init__(self,
                 co_agents: Union[int, list],
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
        super(Where2comm, self).__init__
        
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

        self.layer_num = len(pts_backbone['layers'])
        self.voxel_size = voxel_size
        self.downsample_rate = downsample_rate

        self.voxel_encoder_list = ModuleList([MODELS.build(pts_voxel_encoder) for i in range(self.co_length)])
        self.middle_encoder_list = ModuleList([MODELS.build(pts_middle_encoder) for i in range(self.co_length)])

        # self.voxel_encoder = MODELS.build(pts_voxel_encoder)
        # self.middle_encoder = MODELS.build(pts_middle_encoder)

        self.backbone_list = ModuleList([MODELS.build(pts_backbone) for i in range(self.co_length)])
        # WARNING! Must share the same backbone to ensure rule alignment for feature fusion
        # self.shared_backbone = MODELS.build(pts_backbone)
        
        # Each agent can use its own detection header
        # WARNING! Maybe we should use the same head
        self.detect_list = ModuleList([MODELS.build(pts_detect_module) for i in range(self.co_length)])

        self.shrink = False
        self.shrink_list = []
        if pts_shrink_module:
            self.shrink = True
            self.shrink_list = ModuleList([MODELS.build(pts_shrink_module) for i in range(self.co_length)])
            # WARNING! Must share the same backbone to ensure rule alignment for feature fusion
            # self.shrink_module = MODELS.build(pts_shrink_module)
        
        self.compress = False
        # self.compress_list = []
        if pts_compress_module:
            self.compress = True
            # self.compress_list = ModuleList([MODELS.build(pts_compress_module) for i in range(self.co_length)])
            # WARNING! Must share the same backbone to ensure rule alignment for feature fusion
            self.compress_module = MODELS.build(pts_compress_module)

        self.comm = False
        if pts_comm_module:
            self.comm = True
            self.comm_module = MODELS.build(pts_comm_module)

        self.fusion_list = ModuleList([MODELS.build(pts_fusion_module) for i in range(self.layer_num)])
        # self.fusion_module = MODELS.build(pts_fusion_module)

        # WARNING! The use of independent detection heads can adapt to different nature of final feature inputs
        self.fusion_detect_module = MODELS.build(pts_detect_module)
        self.loss_module = MODELS.build(pts_loss_module)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
    
    def check_fix(self):
        if self.train_cfg and 'fix_cfg' in self.train_cfg: # type: ignore
            fix_cfg = self.train_cfg['fix_cfg']
            if fix_cfg['fix_voxel_encoder']:
                for module in self.voxel_encoder_list:
                    for param in module.parameters():
                        param.requires_grad = False
                # for param in self.voxel_encoder.parameters():
                #     param.requires_grad = False
                print_log("layer voxel_encoder was fixed", "current", logging.WARNING)
            if fix_cfg['fix_middle_encoder']:
                for module in self.middle_encoder_list:
                    for param in module.parameters():
                        param.requires_grad = False
                # for param in self.middle_encoder.parameters():
                #     param.requires_grad = False
                print_log("layer middle_encoder was fixed", "current", logging.WARNING)
            if fix_cfg['fix_shared_backbone']:
                for module in self.backbone_list:
                    for param in module.parameters():
                        param.requires_grad = False
                # for param in self.shared_backbone.parameters():
                #     param.requires_grad = False
                print_log("layer shared_backbone was fixed", "current", logging.WARNING)
            if self.comm and fix_cfg['fix_comm_module']:
                for param in self.comm_module.parameters():
                    param.requires_grad = False
                print_log("layer comm_module was fixed", "current", logging.WARNING)
            if self.shrink and fix_cfg['fix_shrink_module']:
                for module in self.shrink_list:
                    for param in module.parameters():
                        param.requires_grad = False
                # for param in self.shrink_module.parameters():
                #     param.requires_grad = False
                print_log("layer shrink_module was fixed", "current", logging.WARNING)
            if self.compress and fix_cfg['fix_compress_module']:
                for param in self.compress_module.parameters():
                    param.requires_grad = False
                print_log("layer compress_module was fixed", "current", logging.WARNING)
            if fix_cfg['fix_fusion']:
                for module in self.fusion_list:
                    for param in module.parameters():
                        param.requires_grad = False
                print_log("layer fusion_module was fixed", "current", logging.WARNING)
            if fix_cfg['fix_fusion_head']:
                for param in self.fusion_detect_module.parameters():
                    param.requires_grad = False
                print_log("layer fusion_detect_module was fixed", "current", logging.WARNING)
            if fix_cfg['fix_detect_head']:
                for module in self.detect_list:
                    for param in module.parameters():
                        param.requires_grad = False
                print_log("layer detect_module was fixed", "current", logging.WARNING)
        self.check_flag = True

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

        if not self.check_flag:
            self.check_fix()

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

        # scatter_feats = []
        encode_feats = [[] for i in range(self.layer_num)]
        bev_feats = []
        psms = []
        rms = []

        for idx, id in enumerate(self.co_mask):
            pillar_feat = self.voxel_encoder_list[idx](
                batch_dict['voxel_features'][id], # [pls, M, C]
                batch_dict['voxel_num_points'][id], # [pls, ]
                batch_dict['voxel_coords'][id], # [pls, 1 + 3] b z y x
            ) # [pls, C']
            # pillar_feat = self.voxel_encoder(
            #     batch_dict['voxel_features'][id], # [pls, M, C]
            #     batch_dict['voxel_num_points'][id], # [pls, ]
            #     batch_dict['voxel_coords'][id], # [pls, 1 + 3] b z y x
            # ) # [pls, C']
            scatter_feat = self.middle_encoder_list[idx](
                pillar_feat, # [pls, C']
                batch_dict['voxel_coords'][id], # [pls, 1 + 3] b z y x
                batch_size # bs
            ) # [bs, C', H, W]
            # scatter_feat = self.middle_encoder(
            #     pillar_feat, # [pls, C']
            #     batch_dict['voxel_coords'][id], # [pls, 1 + 3] b z y x
            #     batch_size # bs
            # ) # [bs, C', H, W]
            bev_feat = self.backbone_list[idx](
                scatter_feat
            ) # [bs, C'', H, W]
            # bev_feat = self.shared_backbone(
            #     scatter_feat
            # ) # [bs, C'', H, W]
            encode_feat = self.backbone_list[ego_idx](
                scatter_feat,
                True,
                False
            )
            # encode_feat = self.shared_backbone(
            #     scatter_feat,
            #     True,
            #     False
            # )
            if self.shrink:
            #     bev_feat = self.shrink_module(
            #         bev_feat
            #     )
                bev_feat = self.shrink_list[idx](
                    bev_feat
                )
            if self.compress:
                compress_feat = self.compress_module(
                    bev_feat
                )
            psm, rm = self.detect_list[idx](
                bev_feat
            )

            # scatter_feats.append(scatter_feat) # [bs, C', H, W]
            for i in range(self.layer_num):
                encode_feats[i].append(encode_feat[i]) # [bs, C', H, W], [bs, C'*2, H/2, W/2], [bs, C'*4, H/4, W/4]
            bev_feats.append(bev_feat) # [bs, C'', H, W]
            psms.append(psm) # [bs, 2, H, W]
            rms.append(rm) # [bs, 7*2, H, W]
        
        # scatter_feats = torch.stack(scatter_feats, dim=0).transpose(0, 1) # [bs, agents, C', H, W]
        for i in range(self.layer_num):
            encode_feats[i] = torch.stack(encode_feats[i], dim=0).transpose(0, 1) # type: ignore [bs, agents, C', H, W], [bs, agents, C'*2, H/2, W/2], [bs, agents, C'*4, H/4, W/4]
        bev_feats = torch.stack(bev_feats, dim=0).transpose(0, 1) # [bs, agents, C'', H, W]
        psms = torch.stack(psms, dim=0).transpose(0, 1) # [bs, agents, 2, H, W]
        rms = torch.stack(rms, dim=0).transpose(0, 1) # [bs, agents, 7*2, H, W]

        if self.comm:
            communication_masks, ori_communication_maps, communication_maps, communication_rates = \
            self.comm_module(psms, ego_idx)
            # [bs, agents, 1, H, W] in {0, 1}
            # [bs, agents, 1, H, W] in [0, 1]
            # [bs, agents, 1, H, W] in [0, 1] larger
            # [bs, agents, ]

        batch_agent_rela_pose = []
        for b in range(batch_size):
            agent_rela_pose = []
            for idx, id in enumerate(self.co_mask):
                agent_rela_pose.append(torch.linalg.inv(batch_dict['pose_list'][b][id])) # [4, 4]
            batch_agent_rela_pose.append(torch.stack(agent_rela_pose, dim=0)) # [agents, 4, 4]
        batch_agent_rela_pose = torch.stack(batch_agent_rela_pose, dim=0) # [bs, agents, 4, 4]

        x_fuse = []
        for i in range(self.layer_num):
            layer_fuse = []
            for b in range(batch_size):
                rela_pose = batch_agent_rela_pose[b] # [agents, 4, 4]
                agent_feats = encode_feats[i][b] # [agents, C', H, W], [agents, C'*2, H/2, W/2], [agents, C'*4, H/4, W/4]
                if self.comm and i == 0:
                    masked_agent_feats = agent_feats * communication_masks[b] # type: ignore
                else:
                    masked_agent_feats = agent_feats
                
                masked_agent_feats_wrap = warp_affine(masked_agent_feats, # [agents, C', H, W], [agents, C'*2, H/2, W/2], [agents, C'*4, H/4, W/4]
                                    rela_pose, # [agents, 4, 4]
                                    self.downsample_rate,
                                    self.voxel_size)
                
                layer_fuse.append(self.fusion_list[i](masked_agent_feats_wrap, ego_idx))
                # [1, C', H, W], [1, C'*2, H/2, W/2], [1, C'*4, H/4, W/4]
            layer_fuse = torch.stack(layer_fuse, dim=0) # [bs, C', H, W], [bs, C'*2, H/2, W/2], [bs, C'*4, H/4, W/4]
            x_fuse.append(layer_fuse)
        decode_feat = self.backbone_list[ego_idx](x_fuse, False, True) # [bs, C'*2*num_layer, H, W]
        if self.shrink:
            decode_feat = self.shrink_list[ego_idx](
                decode_feat
            ) # [bs, C, H, W]
        # decode_feat = self.shared_backbone(x_fuse, False, True) # [bs, C'*2*num_layer, H, W]
        # if self.shrink:
        #     decode_feat = self.shrink_module(
        #         decode_feat
        #     ) # [bs, C, H, W]

        # fusion_psm, fusion_rm = self.detect_list[ego_idx](decode_feat) # [bs, 2, H, W] # [bs, 2*7, H, W]
        fusion_psm, fusion_rm = self.fusion_detect_module(decode_feat) # [bs, 2, H, W] # [bs, 2*7, H, W]

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
                    'psm': psms[:, idx, ...],
                    'rm': rms[:, idx, ...],
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
            assert self.test_cfg != None
            assert 'agent_threshold' in self.test_cfg # type: ignore
            assert len(self.test_cfg['agent_threshold']) == self.co_length
            assert 'fusion_threshold' in self.test_cfg # type: ignore
            assert 'order' in self.test_cfg # type: ignore
            assert 'nms_threshold' in self.test_cfg # type: ignore
            assert 'lidar_range' in self.test_cfg # type: ignore
            assert 'only_vis' in self.test_cfg

            fusion_corner_pred_3d_final, fusion_masked_cls_pred_final, fusion_masked_gt_corner_final = \
            postprocess(fusion_psm, fusion_rm, self.test_cfg['fusion_threshold'], -1, self.test_cfg, batch_dict) # type: ignore

            corner_pred_3d_final_list = []
            masked_cls_pred_final_list = []
            masked_gt_corner_final_list = []

            for idx, id in enumerate(self.co_mask):
                a, b, c = postprocess(psms[:, idx, ...], rms[:, idx, ...], self.test_cfg['agent_threshold'][idx], id, self.test_cfg, batch_dict) # type: ignore
                corner_pred_3d_final_list.append(a)
                masked_cls_pred_final_list.append(b)
                masked_gt_corner_final_list.append(c)
            
            # # adding dir classifier TODO
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

            global IDX
            START = 13
            END = 23
            fusion_vis_dict = {
                'pc' : batch_dict['vis_pc_list'][0][ego_idx],
                'pc_range' : self.test_cfg['lidar_range'],
                'pred_corner' : fusion_corner_pred_3d_final,
                'gt_corner' : fusion_masked_gt_corner_final,
                'psm' : fusion_psm,
                'feat' : decode_feat
            }
            valid = temp_vis(
                IDX,
                -1,
                fusion_vis_dict,
                False,
                save_start = START,
                save_end = END,
                save_wrap = False
            )
            
            for idx, id in enumerate(self.co_mask):
                vis_dict = {
                    'pc' : batch_dict['vis_pc_list'][0][id],
                    'pc_range' : self.test_cfg['lidar_range'],
                    'pred_corner' : corner_pred_3d_final_list[idx],
                    'gt_corner' : masked_gt_corner_final_list[idx],
                    'cmap' : communication_maps[:, idx, ...], # type: ignore
                    'mask' : communication_masks[:, idx, ...], # type: ignore
                    'psm' : psms[:, idx, ...],
                    'feat' : bev_feats[:, idx, ...],
                    # 'feat' : masked_agent_feats[idx].unsqueeze(0),
                    # 'feat' : masked_agent_feats_wrap[idx].unsqueeze(0),
                    'rela_pose' : batch_agent_rela_pose[:, idx, ...], # type: ignore
                    'downsample_rate' : self.downsample_rate,
                    'voxel_size' : self.voxel_size
                }
                valid = temp_vis(
                    IDX,
                    id,
                    vis_dict,
                    True,
                    save_start = START,
                    save_end = END,
                    save_wrap = True if id != ego_id else False
                )

            if not valid and self.test_cfg['only_vis'] == True:
                import pdb
                pdb.set_trace()
            IDX += 1

            if fusion_corner_pred_3d_final.shape[0] == 0:
                fusion_corner_pred_3d_final = None
                fusion_masked_cls_pred_final = None

            return [fusion_corner_pred_3d_final, fusion_masked_cls_pred_final, fusion_masked_gt_corner_final]
        
        elif mode == 'tensor':

            tensor_list = []
            tensor_list.append(fusion_psm)
            tensor_list.append(fusion_rm)
            for idx, id in enumerate(self.co_mask):
                tensor_list.append(psms[:, idx, ...])
                tensor_list.append(rms[:, idx, ...])

            return tuple(tensor_list)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
        

@MODEL_WRAPPERS.register_module()
class Where2commDDP(DistributedDataParallel):
    def __init__(self,
                 module,
                 detect_anomalous_params: bool = False,
                 **kwargs):
        super().__init__(module=module, **kwargs)
        self.detect_anomalous_params = detect_anomalous_params

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.module.data_preprocessor(data, True) # type: ignore
            losses = self(data, mode='loss')
        parsed_losses, log_vars = self.module.parse_losses(losses) # type: ignore
        optim_wrapper.update_params(parsed_losses)
        if self.detect_anomalous_params:
            detect_anomalous_params(parsed_loss, model=self) # type: ignore
        return log_vars

    def val_step(self, data: Union[dict, tuple, list]) -> list:
        return self.module.val_step(data) # type: ignore

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        return self.module.test_step(data) # type: ignore
