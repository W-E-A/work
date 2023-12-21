# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms import LoadImageFromFile
from pyquaternion import Quaternion

# yapf: disable
from mmdet3d.datasets.transforms import (LoadAnnotations3D,
                                         LoadImageFromFileMono3D,
                                         LoadMultiViewImageFromFiles,
                                         LoadPointsFromFile,
                                         LoadPointsFromMultiSweeps,
                                         MultiScaleFlipAug3D, Pack3DDetInputs,
                                         PointSegClassMapping)
# yapf: enable
from mmdet3d.registry import TRANSFORMS


def is_loading_function(transform):
    """Judge whether a transform function is a loading function.

    Note: `MultiScaleFlipAug3D` is a wrapper for multiple pipeline functions,
    so we need to search if its inner transforms contain any loading function.

    Args:
        transform (dict | :obj:`Pipeline`): A transform config or a function.

    Returns:
        bool: Whether it is a loading function. None means can't judge.
            When transform is `MultiScaleFlipAug3D`, we return None.
    """
    # TODO: use more elegant way to distinguish loading modules
    loading_functions = (LoadImageFromFile, LoadPointsFromFile,
                         LoadAnnotations3D, LoadMultiViewImageFromFiles,
                         LoadPointsFromMultiSweeps, Pack3DDetInputs,
                         LoadImageFromFileMono3D, PointSegClassMapping)
    if isinstance(transform, dict):
        obj_cls = TRANSFORMS.get(transform['type'])
        if obj_cls is None:
            return False
        if obj_cls in loading_functions:
            return True
        if obj_cls in (MultiScaleFlipAug3D, ):
            return None
    elif callable(transform):
        if isinstance(transform, loading_functions):
            return True
        if isinstance(transform, (MultiScaleFlipAug3D)):
            return None
    return False


def get_loading_pipeline(pipeline):
    """Only keep loading image, points and annotations related configuration.

    Args:
        pipeline (list[dict] | list[:obj:`Pipeline`]):
            Data pipeline configs or list of pipeline functions.

    Returns:
        list[dict] | list[:obj:`Pipeline`]): The new pipeline list with only
            keep loading image, points and annotations related configuration.

    Examples:
        >>> transforms = [
        ...    dict(type='LoadPointsFromFile',
        ...         coord_type='LIDAR', load_dim=4, use_dim=4),
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations3D',
        ...         with_bbox=True, with_label_3d=True),
        ...    dict(type='Resize',
        ...         img_scale=[(640, 192), (2560, 768)], keep_ratio=True),
        ...    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
        ...    dict(type='PointsRangeFilter',
        ...         point_cloud_range=point_cloud_range),
        ...    dict(type='ObjectRangeFilter',
        ...         point_cloud_range=point_cloud_range),
        ...    dict(type='PointShuffle'),
        ...    dict(type='Normalize', **img_norm_cfg),
        ...    dict(type='Pad', size_divisor=32),
        ...    dict(type='DefaultFormatBundle3D', class_names=class_names),
        ...    dict(type='Collect3D',
        ...         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadPointsFromFile',
        ...         coord_type='LIDAR', load_dim=4, use_dim=4),
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations3D',
        ...         with_bbox=True, with_label_3d=True),
        ...    dict(type='DefaultFormatBundle3D', class_names=class_names),
        ...    dict(type='Collect3D',
        ...         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
        ...    ]
        >>> assert expected_pipelines == \
        ...        get_loading_pipeline(transforms)
    """
    loading_pipeline = []
    for transform in pipeline:
        is_loading = is_loading_function(transform)
        if is_loading is None:  # MultiScaleFlipAug3D
            # extract its inner pipeline
            if isinstance(transform, dict):
                inner_pipeline = transform.get('transforms', [])
            else:
                inner_pipeline = transform.transforms.transforms
            loading_pipeline.extend(get_loading_pipeline(inner_pipeline))
        elif is_loading:
            loading_pipeline.append(transform)
    assert len(loading_pipeline) > 0, \
        'The data pipeline in your config file must include ' \
        'loading step.'
    return loading_pipeline


def convert_quaternion_to_matrix(quaternion: list,
                                 translation: list = None) -> list: # type: ignore
    """Compute a transform matrix by given quaternion and translation
    vector."""
    result = np.eye(4)
    result[:3, :3] = Quaternion(quaternion).rotation_matrix
    if translation is not None:
        result[:3, 3] = np.array(translation)
    return result.astype(np.float32).tolist()


def tfm_to_pose(tfm):
    """
    Only for dair-v2x-c dataset
    Ruler角内旋系统，ZYX旋转，固定角外旋系统XYZ旋转
    """
    
    # There forumlas are designed from x_to_world, but equal to the one below.
    yaw = np.rad2deg(np.arctan2(tfm[1,0], tfm[0,0])) # clockwise in carla
    roll = np.rad2deg(np.arctan2(-tfm[2,1], tfm[2,2])) # but counter-clockwise in carla
    pitch = np.rad2deg(np.arctan2(tfm[2,0], ((tfm[2,1]**2 + tfm[2,2]**2) ** 0.5)) ) # but counter-clockwise in carla

    x, y, z = tfm[:3,3]
    return(np.array([x, y, z, roll, yaw, pitch]))


def pose_to_tfm(pose):
    """
    Only for dair-v2x-c dataset
    Ruler角内旋系统，ZYX旋转，固定角外旋系统XYZ旋转
    这里的旋转矩阵默认是加负号之后的
    """

    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    tfm = np.identity(4)

    # translation tfm
    tfm[0, 3] = x
    tfm[1, 3] = y
    tfm[2, 3] = z

    # rotation tfm
    tfm[0, 0] = c_p * c_y
    tfm[0, 1] = c_y * s_p * s_r - s_y * c_r
    tfm[0, 2] = -c_y * s_p * c_r - s_y * s_r
    tfm[1, 0] = s_y * c_p
    tfm[1, 1] = s_y * s_p * s_r + c_y * c_r
    tfm[1, 2] = -s_y * s_p * c_r + c_y * s_r
    tfm[2, 0] = s_p
    tfm[2, 1] = -c_p * s_r
    tfm[2, 2] = c_p * c_r

    return tfm

def gen_pose_noise(xy_std,rot_std,xy_mean,rot_mean):
    
    xy = np.random.normal(xy_mean, xy_std, size=(2))
    yaw = np.random.normal(rot_mean, rot_std, size=(1))

    pose_noise = np.array([xy[0], xy[1], 0, 0, yaw[0], 0])

    return pose_noise

def calc_relative_pose(left,right):
    """
    T_AB = T_WA(left)^-1 * T_WB(right)
    """
    left_inv = np.linalg.inv(left)
    if isinstance(right,list):
        result = [left_inv @ item for item in right]
    else:
        result = left_inv @ right
    return result

def corners_to_boxes(corner3d, order='lwh'):
    """
        Z
        ^
        |
        |
        |
        4 -------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1 --------> Y
      |/         |/
      3 -------- 2
     /
    /
   X
    Only for dair-v2x-c dataset

    Convert 8 corners to x, y, z, dx, dy, dz, yaw.
    yaw in radians

    Parameters
    ----------
    corner3d : np.ndarray
        (N, 8, 3)

    order : str, for output.
        'lwh' or 'hwl'

    Returns
    -------
    box3d : np.ndarray
        (N, 7)
    """
    assert corner3d.ndim == 3
    batch_size = corner3d.shape[0]

    xyz = np.mean(corner3d[:, [0, 3, 5, 6], :], axis=1) # (N, 3)
    h = abs(np.mean(corner3d[:, 4:, 2] - corner3d[:, :4, 2], axis=1,
                    keepdims=True)) # (N, 1)
    l = (np.sqrt(np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 5, [0, 1]] - corner3d[:, 6, [0, 1]]) ** 2,
                        axis=1, keepdims=True))) / 4 # (N, 1, 1)

    w = (np.sqrt(
        np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2, axis=1,
               keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 5, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 6, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2,
                        axis=1, keepdims=True))) / 4

    theta = (np.arctan2(corner3d[:, 1, 1] - corner3d[:, 2, 1],
                        corner3d[:, 1, 0] - corner3d[:, 2, 0]) +
             np.arctan2(corner3d[:, 0, 1] - corner3d[:, 3, 1],
                        corner3d[:, 0, 0] - corner3d[:, 3, 0]) +
             np.arctan2(corner3d[:, 5, 1] - corner3d[:, 6, 1],
                        corner3d[:, 5, 0] - corner3d[:, 6, 0]) +
             np.arctan2(corner3d[:, 4, 1] - corner3d[:, 7, 1],
                        corner3d[:, 4, 0] - corner3d[:, 7, 0]))[:,
            np.newaxis] / 4

    if order == 'lwh':
        return np.concatenate([xyz, l, w, h, theta], axis=1).reshape(
            batch_size, 7)
    elif order == 'hwl':
        return np.concatenate([xyz, h, w, l, theta], axis=1).reshape(
            batch_size, 7)
    else:
        raise ValueError(f"order must be 'lwh' or 'hwl', got {order}")

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), radians, angle along z-axis, angle increases x ==> y
    Returns:

    """
    # points, is_numpy = check_numpy_to_torch(points)
    # angle, _ = check_numpy_to_torch(angle)

    cosa = np.cos(angle) # [N, 1]
    sina = np.sin(angle)
    zeros = np.zeros_like(cosa)
    ones = np.ones_like(cosa)
    rot_matrix = np.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros, # warning -sin position here
        zeros, zeros, ones
    ), axis=1).reshape(-1, 3, 3).astype(np.float64) # [N, 3, 3]
    points_rot = np.matmul(points[:, :, 0:3].astype(np.float64), rot_matrix) # reasonable because of above N, 8, 3  8, 3 * 3, 3
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot
# @profile
def boxes_to_corners(boxes, order='lwh'):
    """
        4 -------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2
    Parameters
    __________
    boxes: np.ndarray or torch.Tensor
        (N, 7) [x, y, z, l, w, h, heading], or [x, y, z, h, w, l, heading]

               (x, y, z) is the box center.

    order : str
        'lwh' or 'hwl'

    Returns:
        corners3d: np.ndarray or torch.Tensor
        (N, 8, 3), the 8 corners of the bounding box.


    opv2v's left hand coord 
    
    ^ z
    |
    |
    | . x
    |/
    +-------> y

    """

    boxes_ = boxes

    if order == 'hwl':
        boxes_ = boxes[:, [0, 1, 2, 5, 4, 3, 6]]
    
    template = np.array((
        [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
        [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1],
    )) / 2

    corners3d = np.tile(boxes_[:, None, 3:6], (1, 8, 1))* template[None, :, :] # unrot points8
    corners3d = rotate_points_along_z(corners3d.reshape(-1, 8, 3),boxes_[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes_[:, None, 0:3]

    return corners3d


def mask_gt_outside_range(gt, limit_range, order,
                                   min_num_corners=8, return_mask=False):
    """

    """
    assert gt.shape[1] == 8 or gt.shape[1] == 7

    corners = gt
    if gt.shape[1] == 7:
        corners = boxes_to_corners(gt, order) # [1, 8, 3]

    mask = ((corners >= limit_range[0:3]) &
            (corners <= limit_range[3:6])).all(axis=2)
    mask = mask.sum(axis=1) >= min_num_corners  # (N)

    if return_mask:
        return gt[mask], mask
    return gt[mask]

# @profile
def corner_to_standup_box(box2d):
    """
    Find the minmaxx, minmaxy for each 2d box. (N, 4, 2) -> (N, 4)
    x1, y1, x2, y2

    Parameters
    ----------
    box2d : np.ndarray
        (n, 4, 2), four corners of the 2d bounding box.

    Returns
    -------
    standup_box2d : np.ndarray
        (n, 4)
    """
    N = box2d.shape[0]
    standup_boxes2d = np.zeros((N, 4))

    standup_boxes2d[:, 0] = np.min(box2d[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(box2d[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(box2d[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(box2d[:, :, 1], axis=1)

    return standup_boxes2d