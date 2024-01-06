from typing import Optional, Union, Dict, Tuple, Sequence, List
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule, Sequential
from mmcv.cnn import ConvModule
from ..utils.warper import FeatureWarper
import torch
import torch.nn as nn

@MODELS.register_module()
class TemporalIdentity(BaseModule):
    def __init__(self, position: str = 'last'):
        super(TemporalIdentity, self).__init__()
        assert position in ('first', 'last')
        self.position = position

    def forward(self, x):
        assert x.dim() == 5 # N S C H W
        if self.position == 'last':
            return x[:, -1]
        elif self.position == 'first':
            return x[:, 0]


@MODELS.register_module()
class TemporalNaive(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_cfg: dict = dict(type='Conv2d'),
        norm_cfg: dict = dict(type='BN2d'),
        bias: str = 'auto',
        position: str = 'last',
        with_skip_connect: str = 'none',
        init_cfg: Optional[dict] = None
    ):
        super(TemporalNaive, self).__init__(init_cfg=init_cfg)
        assert position in ('first', 'last')
        self.position = position
        assert with_skip_connect in ('none', 'add', 'cat')
        self.with_skip_connect = with_skip_connect

        if with_skip_connect == 'none':
            inter_channels = max(in_channels // 2, out_channels)
            self.conv = Sequential(
                ConvModule(in_channels, inter_channels, 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg, bias=bias),
                ConvModule(inter_channels, out_channels, 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg, bias=bias)
            )
        elif with_skip_connect == 'add':
            inter_channels = max(in_channels // 2, out_channels)
            self.conv = Sequential(
                ConvModule(in_channels, inter_channels, 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg, bias=bias),
                ConvModule(inter_channels, out_channels, 3, 1, 1, norm_cfg=None, conv_cfg=conv_cfg, bias=bias)
            )
            self.skip = ConvModule(in_channels, out_channels, 1, 1, 1, norm_cfg=None, conv_cfg=conv_cfg, bias=bias)
            self.relu = nn.ReLU(True)
        elif with_skip_connect == 'cat':
            assert out_channels % 2 == 0
            temp_channels = out_channels // 2
            inter_channels = max(in_channels // 2, temp_channels)
            self.conv = Sequential(
                ConvModule(in_channels, inter_channels, 3, 1, 1, norm_cfg=norm_cfg, conv_cfg=conv_cfg, bias=bias),
                ConvModule(inter_channels, temp_channels, 3, 1, 1, norm_cfg=None, conv_cfg=conv_cfg, bias=bias)
            )
            self.skip = ConvModule(in_channels, temp_channels, 1, 1, 1, norm_cfg=None, conv_cfg=conv_cfg, bias=bias)
            self.relu = nn.ReLU(True)

    def forward(self, x):
        assert x.dim() == 5 # N S C H W
        input = x[:, -1] if self.position == 'last' else x[:, 0]
        if self.with_skip_connect == 'none':
            return self.conv(input)
        elif self.with_skip_connect == 'add':
            out1 = self.conv(input)
            out2 = self.skip(input)
            return self.relu(out1 + out2)
        elif self.with_skip_connect == 'cat':
            out1 = self.conv(input)
            out2 = self.skip(input)
            return self.relu(torch.cat((out1, out2), dim=1))
    

# @MODELS.register_module()
# class Temporal3DConvModel(BaseModule):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         history_length: int,
#         out_size_factor: int,
#         grid_size: List[int],
#         extra_channels: int = 0,
#         num_spatial_layers: int = 0,
#         use_pyramid_pooling: bool = True,
#         input_egopose: bool = False,
#         with_skip_connect: bool = False,
#         init_cfg: Optional[dict] = None
#     ):
#         super(Temporal3DConvModel, self).__init__(init_cfg=init_cfg)

#         self.in_channels=in_channels
#         self.out_channels=out_channels
#         self.history_length=history_length
#         self.out_size_factor=out_size_factor
#         self.raw_grid_size=grid_size
#         self.grid_size = [size // out_size_factor for size in self.raw_grid_size]
#         self.extra_channels=extra_channels
#         self.use_pyramid_pooling=use_pyramid_pooling
#         self.input_egopose=input_egopose
#         self.with_skip_connect=with_skip_connect
#         # FIXME


#         self.grid_conf = grid_conf
#         self.spatial_extent = (grid_conf['xbound'][1], grid_conf['ybound'][1])
#         self.receptive_field = receptive_field
#         self.input_egopose = input_egopose

#         self.warper = FeatureWarper(grid_conf=grid_conf)

#         h, w = input_shape
#         modules = []

#         block_in_channels = in_channels
#         block_out_channels = start_out_channels

#         if self.input_egopose:
#             # using 6DoF ego_pose as extra features for input
#             block_in_channels += 6

#         n_temporal_layers = receptive_field - 1
#         for _ in range(n_temporal_layers):
#             if use_pyramid_pooling:
#                 use_pyramid_pooling = True
#                 pool_sizes = [(2, h, w)]
#             else:
#                 use_pyramid_pooling = False
#                 pool_sizes = None
#             temporal = TemporalBlock(
#                 block_in_channels,
#                 block_out_channels,
#                 use_pyramid_pooling=use_pyramid_pooling,
#                 pool_sizes=pool_sizes,
#             )
#             spatial = [
#                 Bottleneck3D(block_out_channels,
#                              block_out_channels, kernel_size=(1, 3, 3))
#                 for _ in range(n_spatial_layers_between_temporal_layers)
#             ]
#             temporal_spatial_layers = nn.Sequential(temporal, *spatial)
#             modules.extend(temporal_spatial_layers)

#             block_in_channels = block_out_channels
#             block_out_channels += extra_in_channels

#         self.model = nn.Sequential(*modules)
#         self.out_channels = block_in_channels
#         self.fp16_enabled = False

#         # skip connection to stablize the present features
#         self.with_skip_connect = with_skip_connect

#     def forward(self, x, future_egomotion, aug_transform=None, img_is_valid=None):
#         input_x = x.clone() # B, S, C, H, W

#         import matplotlib.pyplot as plt
#         import numpy as np
#         def heatmap2d(arr_list):
#             for i, arr in enumerate(arr_list):
#                 arr -= arr.mean()
#                 arr /= arr.std()

#                 # arr = (arr - arr.min()) / (arr.max() - arr.min())

#                 arr *= 64
#                 arr += 128
#                 arr = np.clip(arr, 0, 255).astype('uint8')

#                 plt.figure('Figure %d'%i)
#                 plt.imshow(arr, cmap='viridis')
#                 plt.colorbar()
#             plt.show()
#         # when warping features from temporal frames, the bev-transform should be considered
#         x = self.warper.cumulative_warp_features(
#             x, future_egomotion[:, :x.shape[1]],
#             mode='bilinear', bev_transform=aug_transform,
#         )

#         if self.input_egopose:
#             b, s, _, h, w = x.shape
#             input_future_egomotion = future_egomotion[:, :self.receptive_field].contiguous(
#             )
#             # (2, 3, 6, 128, 128)
#             input_future_egomotion = input_future_egomotion.view(
#                 b, s, -1, 1, 1).expand(b, s, -1, h, w)

#             input_future_egomotion = torch.cat((torch.zeros_like(
#                 input_future_egomotion[:, :1]), input_future_egomotion[:, :-1]), dim=1)

#             # x feature order t - 1, t - 0.5, t
#             x = torch.cat((x, input_future_egomotion), dim=2)

#         # x with shape [b, t, c, h, w]
#         x_valid = img_is_valid[:, :self.receptive_field]
#         for i in range(x.shape[0]):
#             if x_valid[i].all():
#                 continue
#             # pdb.set_trace()
#             invalid_index = torch.where(~x_valid[i])[0][0]
#             valid_feat = x[i, invalid_index + 1]
#             x[i, :(invalid_index + 1)] = valid_feat

#         # Reshape input tensor to (batch, C, time, H, W)
#         x = x.permute(0, 2, 1, 3, 4)
#         x = self.model(x)

#         # x = x.permute(0, 2, 1, 3, 4).contiguous()
#         x = x.permute(0, 2, 1, 3, 4)

#         # both x & input_x have the shape of (batch, time, C, H, W)
#         if self.with_skip_connect:
#             x += input_x

#         # return features of the present frame
#         x = x[:, self.receptive_field - 1]



#         return x