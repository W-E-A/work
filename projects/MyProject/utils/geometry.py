import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from mmdet3d.utils import array_converter
from typing import Tuple, Union, List, Sequence


@array_converter(apply_to=('points', 'extrinsic'))
def simple_points_project(
        points: Union[np.ndarray, Tensor],
        extrinsic: Union[np.ndarray, Tensor],
        ):
    assert points.dim() == 3 # type: ignore
    if extrinsic.dim() == 2: # type: ignore
        extrinsic = extrinsic.unsqueeze(0).tile(points.shape[0], 1, 1) # type: ignore
    else:
        assert extrinsic.dim() == 3 # type: ignore
    device = points.device # type: ignore
    dtype = points.dtype
    point_xyz = points[..., :3]
    cat_info = False
    if points.shape[-1] > 3:
        point_info = points[..., 3:]
        cat_info = True
    point_xyz1 = torch.cat((point_xyz, torch.ones(*point_xyz.shape[:2], 1, dtype=dtype, device=device)), dim=-1) # type: ignore
    result_xyz = torch.einsum("kij, knj->kin", point_xyz1, extrinsic)[..., :3]
    if cat_info:
        result = torch.cat((result_xyz, point_info), dim=-1) # type:ignore
    else:
        result = result_xyz
    return result


def calc_relative_pose(
    left: Union[np.ndarray, Tensor],
    right: Union[np.ndarray, Tensor, List[Union[np.ndarray, Tensor]]],
):
    """
    T_AB = T_WA(left)^-1 * T_WB(right)
    """
    if isinstance(left, np.ndarray):
        left_inv = np.linalg.inv(left)
    elif isinstance(left, Tensor):
        left_inv = torch.linalg.inv(left)
    if isinstance(right, Sequence):
        result = [left_inv @ item for item in right]
    else:
        result = left_inv @ right
    return result


@array_converter(apply_to=('matrix',))
def mat2vec(matrix: Union[np.ndarray, Tensor]):
    """
    Converts a 4x4 pose matrix into a 6-dof pose vector
    Args:
        matrix (ndarray): 4x4 pose matrix
    Returns:
        vector (ndarray): 6-dof pose vector comprising translation components (tx, ty, tz) and
        rotation components (rx, ry, rz)
    """

    # M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    rotx = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2]) # type: ignore

    # M[0, 2] = +siny, M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    cosy = torch.sqrt(matrix[..., 1, 2] ** 2 + matrix[..., 2, 2] ** 2) # type: ignore
    roty = torch.atan2(matrix[..., 0, 2], cosy) # type: ignore

    # M[0, 0] = +cosy*cosz, M[0, 1] = -cosy*sinz
    rotz = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0]) # type: ignore

    rotation = torch.stack((rotx, roty, rotz), dim=-1)

    # Extract translation params
    translation = matrix[..., :3, 3]
    return torch.cat((translation, rotation), dim=-1) # type: ignore


@array_converter(apply_to=('angle',))
def euler2mat(angle: Union[np.ndarray, Tensor]):
    """Convert euler angles to rotation matrix.
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) [Bx3]
    Returns:
        Rotation matrix corresponding to the euler angles [Bx3x3]
    """
    shape = angle.shape
    angle = angle.view(-1, 3) # type: ignore
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)
    zmat = torch.stack([cosz, -sinz, zeros, sinz, cosz, zeros,
                       zeros, zeros, ones], dim=1).view(-1, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny, zeros, ones,
                       zeros, -siny, zeros, cosy], dim=1).view(-1, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros, zeros, cosx, -
                       sinx, zeros, sinx, cosx], dim=1).view(-1, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    rot_mat = rot_mat.view(*shape[:-1], 3, 3)
    return rot_mat


@array_converter(apply_to=('vec',))
def vec2mat(vec: Union[np.ndarray, Tensor]):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz [B,6]
    Returns:
        A transformation matrix [B,4,4]
    """
    # vec N 6
    translation = vec[..., :3].unsqueeze(-1)  # [...x3x1]
    rot = vec[..., 3:].contiguous()  # [...x3]
    rot_mat = euler2mat(rot)  # [...,3,3]
    transform_mat = torch.cat([rot_mat, translation], dim=-1)  # [...,3,4]
    transform_mat = torch.nn.functional.pad(
        transform_mat, [0, 0, 0, 1], value=0)  # [...,4,4]
    transform_mat[..., 3, 3] = 1.0
    return transform_mat


@array_converter(apply_to=('x',))
def invert_pose_matrix(x):
    """
    Parameters
    ----------
        x: [B, 4, 4] batch of pose matrices

    Returns
    -------
        out: [B, 4, 4] batch of inverse pose matrices
    """
    assert len(x.shape) == 3 and x.shape[1:] == (
        4, 4), 'Only works for batch of pose matrices.'

    transposed_rotation = torch.transpose(x[:, :3, :3], 1, 2)
    translation = x[:, :3, 3:]

    inverse_mat = torch.cat(
        [transposed_rotation, -torch.bmm(transposed_rotation, translation)], dim=-1)  # [B,3,4]
    inverse_mat = torch.nn.functional.pad(
        inverse_mat, [0, 0, 0, 1], value=0)  # [B,4,4]
    inverse_mat[..., 3, 3] = 1.0

    return inverse_mat


@array_converter(apply_to=('x', 'flow', ))
def warp_features(
    x: Union[np.ndarray, Tensor],
    flow: Union[np.ndarray, Tensor],
    spatial_extent: Tuple[float, float],
    mode: str = 'nearest',
):
    """ Applies a rotation and translation to feature map x.
    Args:
        x: (b, c, h, w) feature map
        flow: (b, 6) 6DoF vector (only uses the xy poriton)
        mode: use 'nearest' when dealing with categorical inputs
    Returns:
        in plane transformed feature map
    """
    dtype = x.dtype
    device = x.device # type: ignore
    if flow.shape[-1] == 6:
        flow = vec2mat(flow) # B 4 4
    transformation = flow[:, :2, [0, 1, 3]].to(dtype).to(device) # type: ignore
    transformation[:, 0, 2] /= spatial_extent[0] # B 2 3
    transformation[:, 1, 2] /= spatial_extent[1] # B 2 3

    grid = F.affine_grid(
        transformation,
        size=x.shape, # type: ignore
        align_corners=True
    )
    warped_x = F.grid_sample(
        x, # type: ignore
        grid.float(),
        mode=mode,
        padding_mode='zeros',
        align_corners=True
    )

    return warped_x


# def cumulative_warp_features(x, flow, mode='nearest', spatial_extent=None):
#     """ Warps a sequence of feature maps by accumulating incremental 2d flow.

#     x[:, -1] remains unchanged
#     x[:, -2] is warped using flow[:, -2]
#     x[:, -3] is warped using flow[:, -3] @ flow[:, -2]
#     ...
#     x[:, 0] is warped using flow[:, 0] @ ... @ flow[:, -3] @ flow[:, -2]

#     Args:
#         x: (b, t, c, h, w) sequence of feature maps
#         flow: (b, t, 6) sequence of 6 DoF pose
#             from t to t+1 (only uses the xy poriton)

#     """
#     sequence_length = x.shape[1]
#     if sequence_length == 1:
#         return x

#     flow = pose_vec2mat(flow)

#     out = [x[:, -1]]
#     cum_flow = flow[:, -2]
#     for t in reversed(range(sequence_length - 1)):
#         out.append(warp_features(x[:, t], mat2pose_vec(
#             cum_flow), mode=mode, spatial_extent=spatial_extent))
#         # @ is the equivalent of torch.bmm
#         cum_flow = flow[:, t - 1] @ cum_flow

#     return torch.stack(out[::-1], 1)


# def cumulative_warp_features_reverse(x, flow, mode='nearest', spatial_extent=None, bev_transform=None):
#     """ Warps a sequence of feature maps by accumulating incremental 2d flow.

#     x[:, 0] remains unchanged
#     x[:, 1] is warped using flow[:, 0].inverse()
#     x[:, 2] is warped using flow[:, 0].inverse() @ flow[:, 1].inverse()
#     ...

#     Args:
#         x: (b, t, c, h, w) sequence of feature maps
#         flow: (b, t, 6) sequence of 6 DoF pose
#             from t to t+1 (only uses the xy poriton)

#     """

#     flow = pose_vec2mat(flow)
#     out = [x[:, 0]]

#     for i in range(1, x.shape[1]):
#         if i == 1:
#             cum_flow = invert_pose_matrix(flow[:, 0])
#         else:
#             cum_flow = cum_flow @ invert_pose_matrix(flow[:, i - 1])

#         # cum_flow only represents the ego_motion, while bev_transform needs extra processing
#         if bev_transform is not None:
#             # points 先做 inverse_bev_transform，再做 motion 变换，再做 bev_transform
#             # warp_flow = bev_transform @ cum_flow @ bev_transform.inverse()
#             warp_flow = bev_transform.inverse() @ cum_flow @ bev_transform
#         else:
#             warp_flow = cum_flow.clone()

#         out.append(warp_features(x[:, i], mat2pose_vec(
#             warp_flow), mode, spatial_extent=spatial_extent))

#     return torch.stack(out, 1)
