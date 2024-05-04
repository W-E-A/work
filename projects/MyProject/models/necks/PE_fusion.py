from typing import List, Union, Optional
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import build_attention
from mmengine.device import get_device
import torch
import torch.nn as nn
import math


def pos2embed(pos, hidden_dim=128):
    if pos.shape[1] == 3:
        # 这里适配多BEV编码，第一位指示第几个bev，第二位和第三位指示bev上坐标
        pos = pos.permute(2,0,1).contiguous()
    scale = 2 * math.pi
    pos = pos * scale  # H*W, 2
    dim_t = torch.arange(hidden_dim, dtype=torch.float32, device=pos.device)  # hidden_dim 256
    dim_t = 2 * (dim_t // 2) / hidden_dim + 1
    pos_x = pos[..., 0, None] / dim_t  # (H*W, 1) / (hidden_dim,) 广播到同一个维度(H*W, hidden_dim)
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    if pos.shape[1] == 3:
        pos_type = pos[..., 2, None] / dim_t
        pos_type = torch.stack((pos_type[..., 0::2].sin(), pos_type[..., 1::2].cos()), dim=-1).flatten(-2)
        # 输出(H*W, 3*hidden_dim)的正余弦编码
        posemb = torch.cat((pos_type, pos_y, pos_x), dim=-1)
    else:
        posemb = torch.cat((pos_y, pos_x), dim=-1)

    return posemb


@MODELS.register_module()
class Inf2AgentPEFusion(BaseModule):
    def __init__(self,
                 # in_channels: int,
                 # n_head: int,
                 attn_cfgs,
                 mid_channels: Optional[int] = None,
                 # dense_fusion = False,
                 init_cfg: Optional[dict] = None,
                 **kwargs
                 ):
        super().__init__(init_cfg)
        self.hidden_dim = mid_channels
        self.bev_embedding = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.cross_attention = build_attention(attn_cfgs[0])
        pass

    def forward(self, ego_feats, inf_feats, mask):
        '''
        Args:
            ego_feats: (H,W,C) 车端特征
            inf_feats: (H,W,C) 路端特征
            mask: (H,W) bool相关性掩码
        Returns:
        fusion_feats: (H,W,C)
        '''
        inter_inf_feats = inf_feats[mask]  # M,C
        inter_ego_feats = ego_feats[mask]  # M,C
        mask_position = torch.nonzero(mask)  # M,2
        # M,3
        inf_inter_position = torch.cat((mask_position.new_ones(mask_position[:,0:1].shape), mask_position), dim=1)
        ego_inter_position = torch.cat((mask_position.new_zeros(mask_position[:,0:1].shape), mask_position), dim=1)
        # 从(M,3) -> (M,C)位置编码
        inf_pos_embeds = self.bev_embedding(pos2embed(inf_inter_position.to(get_device()), hidden_dim=self.hidden_dim))
        ego_pos_embeds = self.bev_embedding(pos2embed(ego_inter_position.to(get_device()), hidden_dim=self.hidden_dim))
        # query_embedding
        query_pos = self.bev_embedding(pos2embed(mask_position.to(get_device()), hidden_dim=self.hidden_dim))
        # 得到KV和position embedding
        memory = torch.cat([inter_ego_feats, inter_inf_feats], dim=0)
        pos_embed = torch.cat([ego_pos_embeds, inf_pos_embeds], dim=0)

        # cross attention
        key = value = memory
        identity = None
        query = torch.zeros_like(query_pos)
        query_output = self.cross_attention(
            query,
            key,
            value,
            identity,
            query_pos=query_pos,
            key_pos=pos_embed,
            attn_mask=None,
            key_padding_mask=None,
            )

        # output Fusion BEV
        H, W, C = inf_feats.shape
        linear_index = mask_position[:, 0] * W + mask_position[:, 1]
        ego_feats.view(-1, C)[linear_index] = query_output
        pass



