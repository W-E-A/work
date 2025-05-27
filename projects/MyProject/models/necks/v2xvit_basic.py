import math
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
import torch
import torch.nn as nn
from ..fusion import PreNorm, FeedForward, HGTCavAttention, PyramidWindowAttention


class V2XFusionBlock(nn.Module):
    def __init__(self, num_blocks, cav_att_config, pwindow_config):
        super().__init__()
        # first multi-agent attention and then multi-window attention
        self.layers = nn.ModuleList([])
        self.num_blocks = num_blocks

        for _ in range(num_blocks):
            att = HGTCavAttention(cav_att_config['dim'],
                                  heads=cav_att_config['heads'],
                                  dim_head=cav_att_config['dim_head'],
                                  dropout=cav_att_config['dropout']) if \
                cav_att_config['use_hetero'] else \
                CavAttention(cav_att_config['dim'],
                             heads=cav_att_config['heads'],
                             dim_head=cav_att_config['dim_head'],
                             dropout=cav_att_config['dropout'])
            self.layers.append(nn.ModuleList([
                PreNorm(cav_att_config['dim'], att),
                PreNorm(cav_att_config['dim'],
                        PyramidWindowAttention(pwindow_config['dim'],
                                               heads=pwindow_config['heads'],
                                               dim_heads=pwindow_config[
                                                   'dim_head'],
                                               drop_out=pwindow_config[
                                                   'dropout'],
                                               window_size=pwindow_config[
                                                   'window_size'],
                                               relative_pos_embedding=
                                               pwindow_config[
                                                   'relative_pos_embedding'],
                                               fuse_method=pwindow_config[
                                                   'fusion_method']))]))

    def forward(self, x, mask, types):
        for cav_attn, pwindow_attn in self.layers:
            x = cav_attn(x, mask=mask, types=types) + x
            x = pwindow_attn(x) + x
        return x


class V2XTEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        cav_att_config = args['cav_att_config']
        pwindow_att_config = args['pwindow_att_config']
        feed_config = args['feed_forward']

        num_blocks = args['num_blocks']
        depth = args['depth']
        mlp_dim = feed_config['mlp_dim']
        dropout = feed_config['dropout']
        in_channels = args['in_channels']
        
        self.mlp = nn.Linear(in_channels, 256)
        self.mlp2 = nn.Linear(256, 384)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                V2XFusionBlock(num_blocks, cav_att_config, pwindow_att_config),
                PreNorm(cav_att_config['dim'],
                        FeedForward(cav_att_config['dim'], mlp_dim,
                                    dropout=dropout))
            ]))

    def forward(self, ego_feat, infra_feat, com_mask):
        B, C, H, W = ego_feat.shape
        com_mask = com_mask.permute(0,2,3,1).unsqueeze(-1).repeat(1, 1, 1, 1, 2)
        x = torch.stack([ego_feat,infra_feat],dim=1).permute(0,1,3,4,2) #B 2 C H W
        types = torch.stack([torch.zeros(B), torch.ones(B)], dim=1)
        x = self.mlp(x)
        # (B,L,H,W,C)
        for attn, ff in self.layers:
            x = attn(x, mask=com_mask, types=types)
            x = ff(x) + x
        x = self.mlp2(x)
        return x

@MODELS.register_module()
class V2XTransformer(BaseModule):
    def __init__(self, args):
        super(V2XTransformer, self).__init__()

        encoder_args = args['encoder']
        self.encoder = V2XTEncoder(encoder_args)

    def forward(self, ego_feat, infra_feat, mask):
        output = self.encoder(ego_feat, infra_feat, mask)
        output = output[:, 0].permute(0,3,1,2)
        return output