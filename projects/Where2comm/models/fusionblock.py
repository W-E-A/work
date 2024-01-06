import torch
import math
from mmdet3d.registry import MODELS
from mmengine.model.base_module import BaseModule
import torch.nn.functional as F

@MODELS.register_module()
class MaxFusion(BaseModule):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x, ego_idx = None): # [agents, C'', H, W]
        return torch.max(x, dim=0)[0]

@MODELS.register_module()
class PadFusion(BaseModule):
    def __init__(self):
        super(PadFusion, self).__init__()

    def forward(self, x, ego_idx = None): # [agents, C'', H, W]
        mask_1 = torch.sum(x[1], dim=0, keepdim=True) > 1e-5
        mask_2 = torch.sum(x[1], dim=0, keepdim=True) < -1e-5
        mask = torch.logical_or(mask_1, mask_2)
        mask = mask.repeat(x.shape[1], 1, 1)
        feature = x[0]
        feature[mask] = x[1][mask]
        return feature
    
# @MODELS.register_module()
# class SumFusion(BaseModule):
#     def __init__(self):
#         super(SumFusion, self).__init__()

#     def forward(self, x, ego_idx = None): # [agents, C'', H, W]
#         return torch.sum(x, dim=0)

@MODELS.register_module()
class ScaledDotProductAttenFusion(BaseModule):
    def __init__(self):
        super(ScaledDotProductAttenFusion, self).__init__()

    def forward(self, x, ego_idx = None):
        assert ego_idx != None
        cav_num, C, H, W = x.shape # [agents, C'', H, W]
        sqrt_dim = math.sqrt(C)
        x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
        score = torch.bmm(x, x.transpose(1, 2)) / sqrt_dim
        attn = F.softmax(score, dim=-1)
        result = torch.bmm(attn, x) #  (H*W, cav_num, C)
        x = result.permute(1, 2, 0).view(cav_num, C, H, W)[0] # fetrch ego
        return x