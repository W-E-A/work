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
        x = result.permute(1, 2, 0).view(cav_num, C, H, W)[ego_idx] # fetrch ego
        return x