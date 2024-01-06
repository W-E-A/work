import torch.nn as nn
import torch
import numpy as np
from mmdet3d.registry import MODELS
from mmengine.model.base_module import BaseModule

@MODELS.register_module()
class Communication(BaseModule):
    def __init__(self,
                 thres: float = 0.01,
                 smooth: bool = True,
                 smooth_k_size: int = 5,
                 smooth_sigma: float = 1.0):
        super(Communication, self).__init__()
        
        self.thres = thres
        self.smooth = smooth
        self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=smooth_k_size, stride=1, padding=(smooth_k_size-1)//2, bias=False)
        self.init_gaussian_filter(smooth_k_size, smooth_sigma)
        self.gaussian_filter.requires_grad = False # type: ignore
        
    def init_gaussian_filter(self, k_size:int = 5, sigma:float = 1):
        def _gen_gaussian_kernel(k_size:int, sigma:float):
            center = k_size // 2 # 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center] # [5, 5]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)

    def forward(self, agent_psms, ego_idx):

        B, N, C, H, W = agent_psms.shape
        device = agent_psms.device
        psms = agent_psms.reshape(B*N, C, H, W)

        # [bs*agents, 1, H, W]
        ori_communication_maps = torch.max(torch.sigmoid(psms), dim=1, keepdim=True).values # dim1=2 represents the confidence of two anchors
        if self.smooth:
            communication_maps = self.gaussian_filter(ori_communication_maps)
            # [bs*agents, 1, H, W] in [0, 1] larger
        else:
            communication_maps = ori_communication_maps
            # [bs*agents, 1, H, W] in [0, 1] origin

        communication_masks = torch.where(
            communication_maps > self.thres,
            torch.ones_like(communication_maps, device=device),
            torch.zeros_like(communication_maps, device=device)
        )
        # [bs*agents, 1, H, W] in {0, 1}

        communication_rates = torch.sum(communication_masks, dim=(-1, -2, -3)) / communication_masks.size()[-3:].numel()
        # [bs*agents, ]

        # multi ego_idx TODO
        communication_masks[ego_idx::N, ...] = torch.ones((1, H, W), device=device)

        return communication_masks.reshape(B, N, 1, H, W), \
            ori_communication_maps.reshape(B, N, 1, H, W), \
            communication_maps.reshape(B, N, 1, H, W), \
            list(communication_rates)