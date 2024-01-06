# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Implementation of V2VNet Fusion
"""

from turtle import update
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from opencood.models.comm_modules.where2comm import Communication

import pdb

FUSION_CFG = {
    'voxel_size' : [0.4, 0.4, 4],
    'downsample_rate' : 1,
    'agg_operator':{
        'mode': 'ATTEN',
        'feature_dim': 256,
    },
    'num_filters':  [64, 128, 256],
    'layer_nums': [3, 4, 5],
}

COMM_CFG = {
    'thre': 0.01,
    'gaussian_smooth':{
        'k_size': 5,
        'c_sigma': 1.0,
    },
}


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):# [25200, 2, 64]
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim # [25200, 2, 2]
        attn = F.softmax(score, -1) # [25200, 2, 2]
        context = torch.bmm(attn, value) # [25200, 2, 64]
        return context

class AttenFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttenFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape # [2, 64, 100, 252]
        x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x

class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x):
        return torch.max(x, dim=0)[0]

class EncodeLayer(nn.Module):
    def __init__(self, channels, n_head=8, dropout=0):
        super(EncodeLayer, self).__init__()
        self.attn = nn.MultiheadAttention(channels, n_head, dropout)
        self.linear1 = nn.Linear(channels, channels)
        self.linear2 = nn.Linear(channels, channels)

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, q, k, v, confidence_map=None):
        """
        order (seq, batch, feature)
        Args:
            q: (1, H*W, C)
            k: (N, H*W, C)
            v: (N, H*W, C)
        Returns:
            outputs: ()
        """
        residual = q
        if confidence_map is not None:
            context, weight = self.attn(q,k,v, quality_map=confidence_map) # (1, H*W, C)
        else:
            context, weight = self.attn(q,k,v) # (1, H*W, C)
        context = self.dropout1(context)
        output1 = self.norm1(residual + context)

        # feed forward net
        residual = output1 # (1, H*W, C)
        context = self.linear2(self.relu(self.linear1(output1)))
        context = self.dropout2(context)
        output2 = self.norm2(residual + context)

        return output2

class TransformerFusion(nn.Module):
    def __init__(self, channels=256, n_head=8, with_spe=True, with_scm=True, dropout=0):
        super(TransformerFusion, self).__init__()

        self.encode_layer = EncodeLayer(channels, n_head, dropout)
        self.with_spe = with_spe
        self.with_scm = with_scm
        
    def forward(self, batch_neighbor_feature, batch_neighbor_feature_pe, batch_confidence_map, record_len):
        x_fuse = []
        B = len(record_len)
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            neighbor_feature = batch_neighbor_feature[b]
            _, C, H, W = neighbor_feature.shape
            neighbor_feature_flat = neighbor_feature.view(N,C,H*W)  # (N, C, H*W)

            if self.with_spe:
                neighbor_feature_pe = batch_neighbor_feature_pe[b]
                neighbor_feature_flat_pe = neighbor_feature_pe.view(N,C,H*W)  # (N, C, H*W)
                query = neighbor_feature_flat_pe[0:1,...].permute(0,2,1)  # (1, H*W, C)
                key = neighbor_feature_flat_pe.permute(0,2,1)  # (N, H*W, C)
            else:
                query = neighbor_feature_flat[0:1,...].permute(0,2,1)  # (1, H*W, C)
                key = neighbor_feature_flat.permute(0,2,1)  # (N, H*W, C)
            
            value = neighbor_feature_flat.permute(0,2,1)

            if self.with_scm:
                confidence_map = batch_confidence_map[b]
                fused_feature = self.encode_layer(query, key, value, confidence_map)  # (1, H*W, C)
            else:
                fused_feature = self.encode_layer(query, key, value)  # (1, H*W, C)
            
            fused_feature = fused_feature.permute(0,2,1).reshape(1, C, H, W)

            x_fuse.append(fused_feature)
        x_fuse = torch.concat(x_fuse, dim=0)
        return x_fuse

def add_pe_map(x):
    # scale = 2 * math.pi
    temperature = 10000
    num_pos_feats = x.shape[-3] // 2  # positional encoding dimension. C = 2d

    mask = torch.zeros([x.shape[-2], x.shape[-1]], dtype=torch.bool, device=x.device)  #[H, W]
    not_mask = ~mask
    y_embed = not_mask.cumsum(0, dtype=torch.float32)  # [H, W]
    x_embed = not_mask.cumsum(1, dtype=torch.float32)  # [H, W]

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)  # [0,1,2,...,d]
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)  # 10000^(2k/d), k is [0,0,1,1,...,d/2,d/2]

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)  # [C, H, W]

    if len(x.shape) == 4:
        x_pe = x + pos[None,:,:,:]
    elif len(x.shape) == 5:
        x_pe = x + pos[None,None,:,:,:]
    return x_pe # type: ignore


class Where2comm(nn.Module):
    def __init__(self):
        super(Where2comm, self).__init__()

        self.naive_communication = Communication(COMM_CFG)
        self.discrete_ratio = FUSION_CFG['voxel_size'][0]  # voxel_size[0]=0.4    
        self.downsample_rate = FUSION_CFG['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]
        
        self.agg_mode = FUSION_CFG['agg_operator']['mode']

        layer_nums = FUSION_CFG['layer_nums']
        num_filters = FUSION_CFG['num_filters']
        self.num_levels = len(layer_nums)
        self.fuse_modules = nn.ModuleList()
        for idx in range(self.num_levels):
            if self.agg_mode == 'ATTEN':
                fuse_network = AttenFusion(num_filters[idx])
            elif self.agg_mode == 'MAX':
                fuse_network = MaxFusion()
            # elif self.agg_mode == 'Transformer':
            #     fuse_network = TransformerFusion(
            #                                 channels=num_filters[idx], 
            #                                 n_head=args['agg_operator']['n_head'], 
            #                                 with_spe=args['agg_operator']['with_spe'], 
            #                                 with_scm=args['agg_operator']['with_scm'])
            self.fuse_modules.append(fuse_network) # type: ignore

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, rm, record_len, pairwise_t_matrix, backbone=None, heads=None):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, L, 4, 4) 
            
        Returns
        -------
        Fused feature.
        """
        # [8, 64, 200, 504]
        _, C, H, W = x.shape
        # [4, 2, 2, 4, 4]
        B, L = pairwise_t_matrix.shape[:2]

        # (B,L,L,2,3) TODO
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        ups = []
        # backbone.__dict__()
        with_resnet = True if hasattr(backbone, 'resnet') else False
        if with_resnet:
            feats = backbone.resnet(x) # tuple of features # type: ignore
        
        for i in range(self.num_levels):
            x = feats[i] if with_resnet else backbone.blocks[i](x) # type: ignore

            ############ 1. Communication (Mask the features) #########
            if i == 0:
                # [8, 2, 100, 252]
                batch_confidence_maps = self.regroup(rm, record_len)
                _, communication_masks, communication_rates = self.naive_communication(batch_confidence_maps, record_len, pairwise_t_matrix)
                x = x * communication_masks # [8, xx, 100, 252] * [8, 1, 100, 252]
            
            ############ 2. Split the confidence map #######################
            # split x:[(L1, C, H, W), (L2, C, H, W), ...]
            # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
            batch_node_features = self.regroup(x, record_len)
            # 4 * [2, xxx, 100, 252]
            
            ############ 3. Fusion ####################################
            x_fuse = []
            for b in range(B):
                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                node_features = batch_node_features[b]

                # [2, xxx, 100, 252]
                C, H, W = node_features.shape[1:]
                # pdb.set_trace()
                neighbor_feature = warp_affine_simple(node_features, # TODO ??? 
                                                t_matrix[0, :, :, :],
                                                (H, W))
                x_fuse.append(self.fuse_modules[i](neighbor_feature)) # append ego (1, xxx, 100, 252)
            x_fuse = torch.stack(x_fuse) # [4, xxx, 100, 252]

            ############ 4. Deconv ####################################
            if len(backbone.deblocks) > 0: # type: ignore
                ups.append(backbone.deblocks[i](x_fuse)) # type: ignore
            else:
                ups.append(x_fuse)

        if len(ups) > 1:
            x_fuse = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x_fuse = ups[0]
        
        if len(backbone.deblocks) > self.num_levels: # type: ignore
            x_fuse = backbone.deblocks[-1](x_fuse) # type: ignore
        # pdb.set_trace()
        return x_fuse, communication_rates # x_fuse: ego # type: ignore

