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

    def forward(self, agent_psm, ego_idx):

        communication_masks = []
        communication_maps = []
        communication_rates = []

        for idx, psm in enumerate(agent_psm):
            # [bs, 2, H, W]
            ori_communication_map = psm.sigmoid().max(dim=1)[0].unsqueeze(1) # dim1=2 represents the confidence of two anchors
            # [bs, 1, H, W] in [0, 1]
            if self.smooth:
                communication_map = self.gaussian_filter(ori_communication_map)
                # [bs, 1, H, W] in [0, 1] larger
            else:
                communication_map = ori_communication_map
                # [bs, 1, H, W] in [0, 1] origin

            ones_mask = torch.ones_like(communication_map).to(communication_map.device)
            zeros_mask = torch.zeros_like(communication_map).to(communication_map.device)
            communication_mask = torch.where(communication_map > self.thres, ones_mask, zeros_mask)
            # [bs, 1, H, W] in {0,1}

            communication_rate = communication_mask.sum() / communication_mask.numel()

            if idx == ego_idx:
                communication_mask = ones_mask.clone()

            communication_masks.append(communication_mask)
            communication_maps.append(ori_communication_map)
            communication_rates.append(communication_rate) # avg batch comm rate for each agent
        return communication_masks, communication_maps, communication_rates