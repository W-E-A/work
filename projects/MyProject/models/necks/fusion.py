from typing import List, Union, Optional
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
import torch
import torch.nn as nn
from torch import Tensor
import math
import torch.nn.functional as F
from ..fusion import ConvGRU

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


@MODELS.register_module()
class V2VNetFusion(BaseModule):
    def __init__(self,
                 in_channels: int,
                 GRU_H:int,
                 GRU_W:int,
                 GRU_num_layers:int,
                 GRU_kernel_size:List,
                 init_cfg: Optional[dict] = None,
                 **kwargs
                 ):
        super().__init__(init_cfg)

        self.msg_cnn = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3,
                                 stride=1, padding=1)
        self.conv_gru = ConvGRU(input_size=(GRU_H, GRU_W),
                                input_dim=in_channels * 2,
                                hidden_dim=[in_channels],
                                kernel_size=GRU_kernel_size,
                                num_layers=GRU_num_layers,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False)
        self.mlp = nn.Linear(in_channels, in_channels)
       

    def forward(self, ego_feats: Tensor, agent_feats: Tensor, corr_mask: Tensor):
        #使用mask
        B, C, H, W = ego_feats.shape
        all_feats = torch.cat([ego_feats, agent_feats], dim=1) # B C H W
        message = self.msg_cnn(all_feats)
        cat_feature = torch.cat([ego_feats, message], dim=1)
        gru_out = self.conv_gru(cat_feature.unsqueeze(1))[0][0]
        out = gru_out.squeeze(1)
        result = self.mlp(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return result