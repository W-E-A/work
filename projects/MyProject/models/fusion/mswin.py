"""
Multi-scale window transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange


def get_relative_distances(window_size):
    indices = torch.tensor(np.array(
        [[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class BaseWindowAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, drop_out, window_size,
                 relative_pos_embedding):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + \
                                    window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1,
                                                          2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2,
                                                          window_size ** 2))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        b, l, h, w, c, m = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        new_h = h // self.window_size
        new_w = w // self.window_size

        # q : (b, l, m, new_h*new_w, window_size^2, c_head)
        q, k, v = map(
            lambda t: rearrange(t,
                                'b l (new_h w_h) (new_w w_w) (m c) -> b l m (new_h new_w) (w_h w_w) c',
                                m=m, w_h=self.window_size,
                                w_w=self.window_size), qkv)
        # b l m h window_size window_size
        dots = torch.einsum('b l m h i c, b l m h j c -> b l m h i j',
                            q, k, ) * self.scale
        # consider prior knowledge of the local window
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0],
                                       self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b l m h i j, b l m h j c -> b l m h i c', attn, v)
        # b l h w c
        out = rearrange(out,
                        'b l m (new_h new_w) (w_h w_w) c -> b l (new_h w_h) (new_w w_w) (m c)',
                        m=self.heads, w_h=self.window_size,
                        w_w=self.window_size,
                        new_w=new_w, new_h=new_h)
        out = self.to_out(out)

        return out


class PyramidWindowAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads, drop_out, window_size,
                 relative_pos_embedding, fuse_method='naive'):
        super().__init__()

        assert isinstance(window_size, list)
        assert isinstance(heads, list)
        assert isinstance(dim_heads, list)
        assert len(dim_heads) == len(heads)

        self.pwmsa = nn.ModuleList([])

        for (head, dim_head, ws) in zip(heads, dim_heads, window_size):
            self.pwmsa.append(BaseWindowAttention(dim,
                                                  head,
                                                  dim_head,
                                                  drop_out,
                                                  ws,
                                                  relative_pos_embedding))
        self.fuse_mehod = fuse_method
        if fuse_method == 'split_attn':
            self.split_attn = SplitAttn(256)

    def forward(self, x):
        output = None
        # naive fusion will just sum up all window attention output and do a
        # mean
        if self.fuse_mehod == 'naive':
            for wmsa in self.pwmsa:
                output = wmsa(x) if output is None else output + wmsa(x)
            return output / len(self.pwmsa)

        elif self.fuse_mehod == 'split_attn':
            window_list = []
            for wmsa in self.pwmsa:
                window_list.append(wmsa(x))
            return self.split_attn(window_list)

class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        # x: (B, L, 1, 1, 3C)
        batch = x.size(0)
        cav_num = x.size(1)

        if self.radix > 1:
            # x: (B, L, 1, 3, C)
            x = x.view(batch,
                       cav_num,
                       self.cardinality, self.radix, -1)
            x = F.softmax(x, dim=3)
            # B, 3LC
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttn(nn.Module):
    def __init__(self, input_dim):
        super(SplitAttn, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, input_dim, bias=False)
        self.bn1 = nn.LayerNorm(input_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim * 3, bias=False)

        self.rsoftmax = RadixSoftmax(3, 1)

    def forward(self, window_list):
        # window list: [(B, L, H, W, C) * 3]
        assert len(window_list) == 3, 'only 3 windows are supported'

        sw, mw, bw = window_list[0], window_list[1], window_list[2]
        B, L = sw.shape[0], sw.shape[1]

        # global average pooling, B, L, H, W, C
        x_gap = sw + mw + bw
        # B, L, 1, 1, C
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_gap = self.act1(self.bn1(self.fc1(x_gap)))
        # B, L, 1, 1, 3C
        x_attn = self.fc2(x_gap)
        # B L 1 1 3C
        x_attn = self.rsoftmax(x_attn).view(B, L, 1, 1, -1)

        out = sw * x_attn[:, :, :, :, 0:self.input_dim] + \
              mw * x_attn[:, :, :, :, self.input_dim:2*self.input_dim] +\
              bw * x_attn[:, :, :, :, self.input_dim*2:]

        return out