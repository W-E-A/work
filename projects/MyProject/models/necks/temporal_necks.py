from typing import Optional, Union, Dict, Tuple, Sequence, List
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule, Sequential
from mmcv.cnn import ConvModule
import torch
import torch.nn as nn
from ...utils import cumulative_warp_features, FeatureWarper, mat2vec
from ..modules.basic_modules import Bottleneck3D, TemporalBlock


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


@MODELS.register_module()
class Temporal3DConvModel(BaseModule):
    def __init__(
        self,
        pc_range,
        in_channels,
        n_history_and_present,
        input_shape,
        inter_channels=64,
        extra_in_channels=0,
        n_spatial_layers_between_temporal_layers=0,
        use_pyramid_pooling=True,
        input_egopose=False,
        with_skip_connect=False,
        init_cfg=None,
    ):
        super(Temporal3DConvModel, self).__init__(init_cfg)

        self.n_history_and_present = n_history_and_present
        self.input_egopose = input_egopose
        self.warper = FeatureWarper(pc_range=pc_range)

        modules = []

        if self.input_egopose:
            # using 6DoF ego_pose as extra features for input
            in_channels += 6

        n_temporal_layers = n_history_and_present - 1
        for _ in range(n_temporal_layers):
            if use_pyramid_pooling:
                use_pyramid_pooling = True
                pool_sizes = [(2, input_shape[0], input_shape[1])]
            else:
                use_pyramid_pooling = False
                pool_sizes = None
            temporal = TemporalBlock(
                in_channels,
                inter_channels,
                use_pyramid_pooling=use_pyramid_pooling,
                pool_sizes=pool_sizes,
            )
            spatial = [
                Bottleneck3D(inter_channels,
                             inter_channels, kernel_size=(1, 3, 3))
                for _ in range(n_spatial_layers_between_temporal_layers)
            ]
            temporal_spatial_layers = nn.Sequential(temporal, *spatial)
            modules.extend(temporal_spatial_layers)

            in_channels = inter_channels
            inter_channels += extra_in_channels

        self.model = nn.Sequential(*modules)
        # skip connection to stablize the present features
        self.with_skip_connect = with_skip_connect

    def forward(self, x, history_egomotion=None, aug_transform=None):
        input_x = x.clone()
        if history_egomotion:
            # when warping features from temporal frames, the bev-transform should be considered
            x = self.warper.cumulative_warp_features(
                x, history_egomotion,
                mode='bilinear', bev_transform=aug_transform,
            )
        if self.input_egopose:
            b, s, _, h, w = x.shape
            input_history_egomotion = mat2vec(history_egomotion) # b s 4 4 -> b s 6
            input_history_egomotion = input_history_egomotion.view(
                b, s, -1, 1, 1).expand(b, s, -1, h, w)
            input_history_egomotion = torch.cat((torch.zeros_like(
                input_history_egomotion[:, :1]), input_history_egomotion[:, :-1]), dim=1) # 除了倒数第一位之外的有效
            x = torch.cat((x, input_history_egomotion), dim=2)

        # Reshape input tensor to (batch, C, time, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # both x & input_x have the shape of (batch, time, C, H, W)
        if self.with_skip_connect:
            x += input_x

        # return features of the present frame
        x = x[:, self.n_history_and_present - 1]

        return x