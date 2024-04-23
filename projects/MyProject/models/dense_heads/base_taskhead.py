import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

import pdb


@MODELS.register_module()
class BaseTaskHead(BaseModule):
    def __init__(self, task_dict, in_channels, inter_channels=None,
                 init_cfg=dict(type='Kaiming', layer='Conv2d'),
                 norm_cfg=dict(type='BN'),
                 ):
        super(BaseTaskHead, self).__init__(init_cfg=init_cfg)

        self.task_heads = nn.ModuleDict()
        inter_channels = in_channels if inter_channels is None else inter_channels
        for task_key, task_dim in task_dict.items():
            self.task_heads[task_key] = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, task_dim, kernel_size=1, padding=0)
            )

        self.fp16_enabled = False

    def loss(self, x):
        pass

    def forward(self, x, targets=None):
        self.float()
        x = x[0]
        return {task_key: task_head(x) for task_key, task_head in self.task_heads.items()}
