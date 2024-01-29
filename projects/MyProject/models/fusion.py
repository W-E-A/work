from typing import List, Union, Optional
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
import torch
import torch.nn as nn
from torch import Tensor

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

    def forward(self, ego_feats: Tensor, agent_feats: Tensor):
        B, E, C, H, W = ego_feats.shape
        _, A, _, _, _ = agent_feats.shape
        # FIXME if dense fusion
        ego_feats = ego_feats.permute(0, 3, 4, 1, 2).contiguous().view(B*H*W, E, C) # N E C
        agent_feats = agent_feats.permute(0, 3, 4, 1, 2).contiguous().view(B*H*W, A, C) # N A C
        all_feats = torch.cat([ego_feats, agent_feats], dim=1) # N A+E C
        memory = self.encoder(all_feats) # N A+E C
        result = self.decoder(ego_feats, memory) # N E C
        result = result.view(B, H, W, E, C).permute(0, 3, 4, 1, 2).contiguous() # B E C H W
        return result