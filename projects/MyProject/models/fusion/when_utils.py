import torch.nn as nn
import torch
from mmdet3d.registry import MODELS
from mmengine.model.base_module import BaseModule

@MODELS.register_module()
class linear(BaseModule):
    def __init__(self, out_size=128, in_channels=256, input_feat_sz=256):
        super(linear, self).__init__()
        feat_map_sz = input_feat_sz // 4
        self.n_feat = int(in_channels * feat_map_sz * feat_map_sz)

        self.fc = nn.Sequential(
            nn.Linear(self.n_feat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_size)
        )

    def forward(self, features_map):
        outputs = self.fc(features_map.view(-1, self.n_feat))
        return outputs

@MODELS.register_module()
class conv2DBatchNormRelu(BaseModule):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

@MODELS.register_module()
class policy_net4(BaseModule):
    def __init__(self, in_channels=256):
        super(policy_net4, self).__init__()
        self.in_channels = in_channels

        # Encoder
        # down 1 
        self.conv1 = conv2DBatchNormRelu(self.in_channels, 256, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(256, 128, k_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(128, 128, k_size=3, stride=2, padding=1)

        # down 2
        self.conv4 = conv2DBatchNormRelu(128, 128, k_size=3, stride=1, padding=1)
        self.conv5 = conv2DBatchNormRelu(128, 128, k_size=3, stride=2, padding=1)

    def forward(self, features_map):
        outputs = self.conv1(features_map)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        return outputs

@MODELS.register_module()
class GeneralDotProductAttention(BaseModule):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, query_size, key_size, attn_dropout=0.1):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(query_size, key_size)
        print('Msg size: ',query_size,'  Key size: ', key_size)

    def forward(self, q, k, v):
        # q (batch,1,128)
        # k (batch,2,128)
        # v (batch,2,channel*size*size)
        query = self.linear(q)  # (batch,1,key_size)
        attn_orig = torch.bmm(k, query.transpose(2, 1))  # (batch,2,1)
        attn_orig = self.softmax(attn_orig)  # (batch,2,1)
        attn = torch.unsqueeze(torch.unsqueeze(attn_orig, 3), 4)  # (batch,2,1,1,1)
        output = attn * v  # (batch,2,channel,size,size)
        output = output.sum(1)  # (batch,1,channel,size,size)
        return output