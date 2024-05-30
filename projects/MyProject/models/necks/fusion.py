from typing import List, Union, Optional
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
import torch
import torch.nn as nn
from torch import Tensor
import math
import torch.nn.functional as F

@MODELS.register_module()
class V2XTransformerFusion(BaseModule):
    def __init__(self,
                 in_channels: int,
                 n_head: int,
                 mid_channels: Optional[int] = None,
                 dense_fusion: bool = True,
                 init_cfg: Optional[dict] = None,
                 **kwargs
                 ):
        super().__init__(init_cfg)
        self.dense_fusion = dense_fusion
        mid_channels = in_channels // 2 if not mid_channels else mid_channels
        self.encoder = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=n_head,
            dim_feedforward=mid_channels,
            batch_first=True,
            **kwargs,
        )
        self.decoder = nn.TransformerDecoderLayer(
            d_model=in_channels,
            nhead=n_head,
            dim_feedforward=mid_channels,
            batch_first=True,
            **kwargs,
        )

    def forward(self, ego_feats: Tensor, agent_feats: Tensor, corr_mask: Tensor):
        #使用mask
        B, C, H, W = ego_feats.shape
        ego_feats_fusion = ego_feats.permute(0, 2, 3, 1).contiguous().view(B*H*W, 1, C) # N E C
        agent_feats = agent_feats.permute(0, 2, 3, 1).contiguous().view(B*H*W, 1, C) # N A C
        all_feats = torch.cat([ego_feats_fusion, agent_feats], dim=1) # N A+E C

        #ScaledDotProduct 29.7
        sqrt_dim = math.sqrt(C)
        score = torch.bmm(all_feats, all_feats.transpose(1, 2)) / sqrt_dim
        attn = F.softmax(score, dim=-1)
        result = torch.bmm(attn, all_feats) #  N A+E C
        result = result[:,0:1,:].view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        result = result*corr_mask.float() + ego_feats*(~corr_mask).float()
        return result

        # B, C, H, W = ego_feats.shape
        # ego_feats = ego_feats.permute(0, 2, 3, 1).contiguous().view(B*H*W, 1, C) # N E C
        # agent_feats = agent_feats.permute(0, 2, 3, 1).contiguous().view(B*H*W, 1, C) # N A C
        # all_feats = torch.cat([ego_feats, agent_feats], dim=1) # N A+E C

        # #ScaledDotProduct 29.7
        # sqrt_dim = math.sqrt(C)
        # score = torch.bmm(all_feats, all_feats.transpose(1, 2)) / sqrt_dim
        # attn = F.softmax(score, dim=-1)
        # result = torch.bmm(attn, all_feats) #  N A+E C
        # result = result[:,0:1,:].view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        # return result

        #ScaledDotProductSum 27
        # sqrt_dim = math.sqrt(C)
        # score = torch.bmm(all_feats, all_feats.transpose(1, 2)) / sqrt_dim
        # attn = F.softmax(score, dim=-1)
        # result = torch.bmm(attn, all_feats) #  N A+E C
        # result = torch.sum(result, dim=1, keepdim=True)
        # result = result.view(B, H, W, E, C).permute(0, 3, 4, 1, 2).contiguous()
        # return result
        
        #Sum 25
        # result = torch.sum(all_feats, dim=1, keepdim=True)
        # result = result.view(B, H, W, E, C).permute(0, 3, 4, 1, 2).contiguous() # B E C H W
        # return result

        #Origin Revise1
        # memory = self.encoder(all_feats) # N A+E C
        # result = torch.sum(memory, dim=1, keepdim=True)
        # result = result.view(B, H, W, E, C).permute(0, 3, 4, 1, 2).contiguous() # B E C H W
        # return result

        # Origin Revise2 27
        # result = self.encoder(all_feats) # N A+E C
        # result = result[:,0:1,:].view(B, H, W, E, C).permute(0, 3, 4, 1, 2).contiguous()
        # return result

        #Origin 20
        # memory = self.encoder(all_feats) # N A+E C
        # result = self.decoder(ego_feats, memory) # N E C
        # result = result.view(B, H, W, E, C).permute(0, 3, 4, 1, 2).contiguous() # B E C H W
        # return result