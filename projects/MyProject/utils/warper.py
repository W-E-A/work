import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from .geometry import vec2mat, invert_pose_matrix

class FeatureWarper(object):
    def __init__(self, pc_range):
        self.pc_range = pc_range
        self.spatial_extent = (self.pc_range[3], self.pc_range[4])

    def warp_features(
        self,
        x: Union[np.ndarray, Tensor],
        flow: Union[np.ndarray, Tensor],
        mode: str = 'nearest',
        ):
        """
        Applies a rotation and translation to feature map x.
        
        follow: P_A = torch(T_{BA})T_B

        Args:
            x: (b, c, h, w) feature map
            flow: (b, 6) 6DoF vector (only uses the xy poriton)
            mode: use 'nearest' when dealing with categorical inputs
        Returns:
            in plane transformed feature map
        """
        dtype = x.dtype
        device = x.device # type: ignore
        shape = x.shape
        if flow.shape[-1] == 6:
            flow = vec2mat(flow) # B 4 4

        transformation = flow[:, :2, [0, 1, 3]].to(dtype).to(device) # type: ignore
        transformation[:, 0, 2] /= self.spatial_extent[0] # B 2 3
        transformation[:, 1, 2] /= self.spatial_extent[1] # B 2 3

        grid = F.affine_grid(
            transformation,
            size=shape, # type: ignore
            align_corners=True
        ).to(dtype)
        warped_x = F.grid_sample(
            x, # type: ignore
            grid,
            mode=mode,
            padding_mode='zeros',
            align_corners=True
        )

        return warped_x
    
    def cumulative_warp_features(self, x, flow, mode='nearest'):
        """ Warps a sequence of feature maps by accumulating incremental 2d flow.
        T_{01} T_{12} T_{23} T_{34} ...T_{n-1n} I
                                        -2   -1
        如果特征图要聚合到最后一帧也就是历史聚合到现在，则用此函数，并按照上述格式制作flow

        x[:, -1] remains unchanged
        x[:, -2] is warped using flow[:, -2]
        x[:, -3] is warped using flow[:, -3] @ flow[:, -2]
        ...
        x[:, 0] is warped using flow[:, 0] @ ... @ flow[:, -3] @ flow[:, -2]

        Args:
            x: (b, t, c, h, w) sequence of feature maps
            flow: (b, t, 6) sequence of 6 DoF pose
                from t to t+1 (only uses the xy poriton)

        """
        sequence_length = x.shape[1]
        if sequence_length == 1:
            return x

        if flow.shape[-1] == 6:
            flow = vec2mat(flow) # B 4 4

        out = [x[:, -1]]
        cum_flow = flow[:, -2]
        for t in reversed(range(sequence_length - 1)):
            out.append(
                self.warp_features(
                    x[:, t],
                    cum_flow,
                    mode=mode)
                )
            # @ is the equivalent of torch.bmm
            cum_flow = flow[:, t - 1] @ cum_flow

        return torch.stack(out[::-1], 1)

    def cumulative_warp_features_reverse(self, x, flow, mode='nearest', bev_transform=None):
        """ Warps a sequence of feature maps by accumulating incremental 2d flow.
        I T_{10} T_{21} T_{32} ... T_{nn-1}
        0    1    2
        如果特征图要聚合到第一帧也就是未来聚合到现在，则用此函数，并按照上述格式制作flow

        x[:, 0] remains unchanged
        x[:, 1] is warped using flow[:, 0].inverse()
        x[:, 2] is warped using flow[:, 0].inverse() @ flow[:, 1].inverse()
        ...

        Args:
            x: (b, t, c, h, w) sequence of feature maps
            flow: (b, t, 6) sequence of 6 DoF pose
                from t to t+1 (only uses the xy poriton)

        """

        if flow.shape[-1] == 6:
            flow = vec2mat(flow)
        out = [x[:, 0]]

        for i in range(1, x.shape[1]):
            if i == 1:
                cum_flow = invert_pose_matrix(flow[:, 1])
            else:
                cum_flow = cum_flow @ invert_pose_matrix(flow[:, i])
            
            # cum_flow only represents the ego_motion, while bev_transform needs extra processing
            if bev_transform is not None:
                # points 先做 inverse_bev_transform，再做 motion 变换，再做 bev_transform
                warp_flow = bev_transform @ cum_flow @ bev_transform.inverse()
            else:
                warp_flow = cum_flow.clone()

            out.append(
                self.warp_features(
                    x[:, i],
                    invert_pose_matrix(warp_flow), # necessary
                    mode=mode)
                )

        return torch.stack(out, 1)

    def cumulative_warp_features_single(self, x, flow, mode='nearest', bev_transform=None):
        """
        follow: P_A = T_{AB}P_B
        """

        if flow.shape[-1] == 6:
            flow = vec2mat(flow)
        
        flow = invert_pose_matrix(flow)

        if bev_transform is not None:
            # points 先做 inverse_bev_transform，再做 motion 变换，再做 bev_transform
            warp_flow = bev_transform @ flow @ bev_transform.inverse()
        else:
            warp_flow = flow.clone()

        out = self.warp_features(
            x, 
            warp_flow, 
            mode
        )

        return out

    def cumulative_warp_features_batch(self, x, flow, mode='nearest', bev_transform=None):
        """
        follow: P_A = T_{AB}P_B
        """
        out = []
        for i in range(x.shape[1]):
            out.append(self.cumulative_warp_features_single(x[:, i], flow[:, i]))
        return torch.stack(out, 1)

    def cumulative_warp_features_v2x(self, x, flow, mode='nearest', bev_transform=None):
        """
        follow: P_A = T_{AB}P_B
        """
        out = [x[:, 0]] # 这里默认第一个是ego FIXME

        for i in range(1, x.shape[1]): # 对于每个代理转到ego下需要输入的flow是T_{ego - agent}
            out.append(self.cumulative_warp_features_single(x[:, i], flow[:, i]))

        return torch.stack(out, 1)
