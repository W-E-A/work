from typing import List, Union, Optional
from mmdet3d.registry import MODELS
from mmengine.model.base_module import BaseModule
import torch
import torch.nn as nn
from .resblock import resnet_modified_basic


@MODELS.register_module()
class BEVBackbone(BaseModule):
    def __init__(self,
                 layers: list = [], # [3, 4, 5]
                 layer_strides: list = [], # [2, 2, 2]
                 num_filters: list = [], # [64, 128, 256]
                 upsample_strides: list = [],  # [1, 2, 4]
                 num_upsample_filters: list = [], # [128, 128, 128]
                 init_cfg: Union[dict, List[dict], None] = None):
        super(BEVBackbone, self).__init__(init_cfg)
        
        assert len(layers) == len(num_filters) == len(layer_strides)
        assert len(num_upsample_filters) == len(upsample_strides)
        assert len(layers) > 0
        if len(upsample_strides) > 0:
            assert len(upsample_strides) >= len(layers)

        self.resnet = resnet_modified_basic(layers, layer_strides, num_filters)

        self.num_levels = len(layers)
        self.deblocks = []
        
        for idx in range(self.num_levels):
            if len(upsample_strides) > 0:
                self.deblocks.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx], upsample_strides[idx], upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True)
                    )
                )
        self.num_bev_feats = sum(num_upsample_filters) # [384]
        if len(upsample_strides) > self.num_levels:
            self.deblocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.num_bev_feats, self.num_bev_feats, upsample_strides[-1], upsample_strides[-1], bias=False),
                    nn.BatchNorm2d(self.num_bev_feats, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.deblocks = nn.ModuleList(self.deblocks)
        
    def forward(self, x, encode: bool = True, decode :bool = True, decode_level: Optional[Union[int, tuple]] = None):

        if encode:
            x = self.resnet(x)

        if not decode:
            # [B, 64, H, W]
            return x
        
        if decode_level == None:
            decode_level = tuple(range(self.num_levels))
        
        if not encode:
            if isinstance(decode_level, int):
                assert decode_level < self.num_levels
                return self.deblocks[decode_level](x)
            elif isinstance(decode_level, tuple):
                assert isinstance(x, list) and len(x) == len(decode_level)
                assert all(level in list(range(self.num_levels)) for level in decode_level)
        
        ups = []
        for idx in decode_level: # type: ignore
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[idx](x[idx]))
            else:
                ups.append(x[idx])

        if len(self.deblocks) > 0:
            x = torch.cat(ups, dim=1)
        
        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)

        return x
