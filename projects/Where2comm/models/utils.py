import torch
import torch.nn as nn
import numpy as np
from mmdet3d.registry import MODELS
from mmengine.model.base_module import BaseModule
from mmdet3d.models.middle_encoders.pillar_scatter import PointPillarsScatter
import torch.nn.functional as F
import math

from shapely.geometry import Polygon

@MODELS.register_module()
class PointPillarsScatterWrapper(BaseModule):
    def __init__(self,
                 in_channels: int,
                 lidar_range: list,
                 voxel_size: list):
        super(PointPillarsScatterWrapper, self).__init__()
        self.in_channels = in_channels
        self.lidar_range = lidar_range
        self.voxel_size = voxel_size

        D = math.ceil((lidar_range[5] - lidar_range[2]) / voxel_size[2])
        H = math.ceil((lidar_range[4] - lidar_range[1]) / voxel_size[1])
        W = math.ceil((lidar_range[3] - lidar_range[0]) / voxel_size[0])

        self.scatter = PointPillarsScatter(in_channels, [H, W])
    
    def forward(self, *args, **kwargs):
        return self.scatter( *args, **kwargs)


@MODELS.register_module()
class ShrinkModule(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shrink_step: int = 1,
                 norm: bool = False):
        super(ShrinkModule, self).__init__()

        assert out_channels < in_channels
        shrink_delta = (in_channels - out_channels) // shrink_step
        shrink_channels = [in_channels - shrink_delta * i for i in range(shrink_step + 1)]
        if (in_channels - out_channels) % shrink_step != 0:
            shrink_channels.append(out_channels)
        
        self.shrink_layers = []
        for i in range(len(shrink_channels) - 1):
            c_in = shrink_channels[i]
            c_out = shrink_channels[i+1]
            if norm:
                self.shrink_layers.append(
                    nn.Sequential(
                        nn.Conv2d(c_in, c_out, 1, bias=False),
                        nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                        nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                self.shrink_layers.append(
                    nn.Sequential(
                        nn.Conv2d(c_in, c_out, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(c_out, c_out, 3, padding=1),
                        nn.ReLU(inplace=True)
                    )
                )
        self.shrink_layers = nn.ModuleList(self.shrink_layers)
        
    def forward(self, x):
        for layer in self.shrink_layers:
            x = layer(x)
        
        return x

@MODELS.register_module()
class CompressModule(BaseModule):
    def __init__(self,
                 in_channels: int,
                 compress_ratio):
        super(CompressModule, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//compress_ratio, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(in_channels//compress_ratio, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels//compress_ratio, in_channels, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels, eps=1e-3,
                           momentum=0.01),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

@MODELS.register_module()
class DetectHead(BaseModule):
    def __init__(self,
                 in_channels: int,
                 num_anchors: int = 2):
        super(DetectHead, self).__init__()

        self.cls_head = nn.Conv2d(in_channels, num_anchors, 1)
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 7, 1)

    def forward(self, x):
        psm = self.cls_head(x)
        rm = self.reg_head(x)

        return psm, rm
    

def warp_affine(feature, tfm, downsample_rate, voxel_size, align_corners: bool = False):
    # feature [agents, C'', H, W]
    # recover to origin raw voxel feature size by ' * downsample_rate ', can adept to multi scale
    H = feature.shape[-2] * downsample_rate
    W = feature.shape[-1] * downsample_rate
    
    tfm = tfm[:, :2, [0, 1, 3]] # [2, 3]
    tfm[:, 0, 1] = tfm[:, 0, 1] * H / W
    tfm[:, 1, 0] = tfm[:, 1, 0] * W / H
    tfm[:, 0, 2] = tfm[:, 0, 2] / (voxel_size[0] * W) * 2
    tfm[:, 1, 2] = tfm[:, 1, 2] / (voxel_size[1] * H) * 2

    tfm = tfm.to(dtype=feature.dtype)
    grid = F.affine_grid(
        tfm, # origin trans
        list(feature.size()), # but adept to feature size
        align_corners=align_corners
    ).to(feature.device)
    return F.grid_sample(feature, grid, align_corners=align_corners)


def decode_reg_result(reg_result, anchor_box):

    batch_size = reg_result.shape[0]

    reg_result = reg_result.permute(0, 2, 3, 1).contiguous().reshape(batch_size, -1, 7) # [B, H*W*2, 7]
    boxes = torch.zeros_like(reg_result) # [B, H*W*2, 7]
    
    anchor_box = anchor_box.reshape(-1, 7).repeat(batch_size, 1, 1).to(reg_result.dtype) # [B, H*W*2, 7]
    anchor_d = torch.sqrt(anchor_box[..., 4] ** 2 + anchor_box[..., 5] ** 2) # [h*w*2, 1]
    anchor_d = anchor_d.repeat(batch_size, 2, 1).transpose(1, 2) # [B, H*W*2, 2]

    # import pdb
    # pdb.set_trace()

    # Inv-normalize to get xyz
    boxes[..., [0, 1]] = torch.mul(reg_result[..., [0, 1]], anchor_d) + anchor_box[..., [0, 1]]
    boxes[..., [2]] = torch.mul(reg_result[..., [2]], anchor_box[..., [3]]) + anchor_box[..., [2]]

    # hwl
    boxes[..., [3, 4, 5]] = torch.exp(reg_result[..., [3, 4, 5]]) * anchor_box[..., [3, 4, 5]]
    # yaw angle
    boxes[..., 6] = reg_result[..., 6] + anchor_box[..., 6]

    return boxes


def get_3d_8points(obj_size, yaw_lidar, center_lidar):
        # yaw_lidar = -yaw_lidar
        lidar_r = torch.tensor(
            [
                [math.cos(yaw_lidar), -math.sin(yaw_lidar), 0],
                [math.sin(yaw_lidar), math.cos(yaw_lidar), 0],
                [0, 0, 1],
            ] # type: ignore
        )
        l, w, h = obj_size
        corners_3d_lidar = torch.tensor(
            [
                [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                [0, 0, 0, 0, h, h, h, h],
            ]
        )
        # import pdb
        # pdb.set_trace()

        corners_3d_lidar = lidar_r @ corners_3d_lidar + torch.tensor(center_lidar).unsqueeze(-1)
        return corners_3d_lidar.T


def boxes_to_corners_baseline(boxes, order='lwh'):
    corners_list = []
    if order == 'hwl':
        boxes = boxes[:, [0, 1, 2, 5, 4, 3, 6]]
    for idx in range(boxes.shape[0]):
        box = boxes[idx]
        x, y, z, l, w, h, yaw = box
        # import pdb
        # pdb.set_trace()
        obj_size, yaw_lidar, center_lidar = \
        [l.item(), w.item(), h.item()], yaw.item(), [x.item(), y.item(), z.item() - h.item()/2]
        corners = get_3d_8points(obj_size, yaw_lidar, center_lidar)
        corners_list.append(corners[np.newaxis, :, :])
    return torch.concat(corners_list, dim=0)

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
    standup_boxes2d = torch.zeros((N, 4))
    standup_boxes2d = standup_boxes2d.to(box2d.device)

    standup_boxes2d[:, 0] = torch.min(box2d[:, :, 0], dim=1).values
    standup_boxes2d[:, 1] = torch.min(box2d[:, :, 1], dim=1).values
    standup_boxes2d[:, 2] = torch.max(box2d[:, :, 0], dim=1).values
    standup_boxes2d[:, 3] = torch.max(box2d[:, :, 1], dim=1).values

    return standup_boxes2d

def remove_large_pred_bbx(bbx_3d):
    """
    Remove large bounding box.

    Parameters
    ----------
    bbx_3d : torch.Tensor
        Predcited 3d bounding box, shape:(N,8,3)

    Returns
    -------
    index : torch.Tensor
        The keep index.
    """
    bbx_x_max = torch.max(bbx_3d[:, :, 0], dim=1)[0]
    bbx_x_min = torch.min(bbx_3d[:, :, 0], dim=1)[0]
    x_len = bbx_x_max - bbx_x_min

    bbx_y_max = torch.max(bbx_3d[:, :, 1], dim=1)[0]
    bbx_y_min = torch.min(bbx_3d[:, :, 1], dim=1)[0]
    y_len = bbx_y_max - bbx_y_min

    bbx_z_max = torch.max(bbx_3d[:, :, 1], dim=1)[0]
    bbx_z_min = torch.min(bbx_3d[:, :, 1], dim=1)[0]
    z_len = bbx_z_max - bbx_z_min

    index = torch.logical_and(x_len <= 6, y_len <= 6)
    index = torch.logical_and(index, z_len)

    return index

def remove_bbx_abnormal_z(bbx_3d):
    """
    Remove bounding box that has negative z axis.

    Parameters
    ----------
    bbx_3d : torch.Tensor
        Predcited 3d bounding box, shape:(N,8,3)

    Returns
    -------
    index : torch.Tensor
        The keep index.
    """
    bbx_z_min = torch.min(bbx_3d[:, :, 2], dim=1)[0]
    bbx_z_max = torch.max(bbx_3d[:, :, 2], dim=1)[0]
    index = torch.logical_and(bbx_z_min >= -3, bbx_z_max <= 1)

    return index

def nms_rotated(boxes, scores, threshold):
    """Performs rorated non-maximum suppression and returns indices of kept
    boxes.

    Parameters
    ----------
    boxes : torch.tensor
        The location preds with shape (N, 4, 2).

    scores : torch.tensor
        The predicted confidence score with shape (N,)

    threshold: float
        IoU threshold to use for filtering.

    Returns
    -------
        An array of index
    """
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)
    boxes = boxes.cpu().detach().numpy()
    scores = scores.cpu().detach().numpy()

    polygons = convert_format(boxes)

    top = 1000
    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1][:top]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(polygons[i], polygons[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    return np.array(pick, dtype=np.int32)

def convert_format(boxes_array):
    """
    Convert boxes array to shapely.geometry.Polygon format.
    Parameters
    ----------
    boxes_array : np.ndarray
        (N, 4, 2) or (N, 8, 3).
    Returns
    -------
        list of converted shapely.geometry.Polygon object.
    """
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in
                boxes_array]
    return np.array(polygons)

def compute_iou(box, boxes):
    """
    Compute iou between box and boxes list
    Parameters
    ----------
    box : shapely.geometry.Polygon
        Bounding box Polygon.

    boxes : list
        List of shapely.geometry.Polygon.

    Returns
    -------
    iou : np.ndarray
        Array of iou between box and boxes.

    """
    # Calculate intersection areas
    if np.any(np.array([box.union(b).area for b in boxes])==0):
        print('debug')
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]

    return np.array(iou, dtype=np.float32)

def get_mask_for_boxes_within_range(boxes, gt_range):
    """
    Generate mask to remove the bounding boxes
    outside the range.

    Parameters
    ----------
    boxes : torch.Tensor
        Groundtruth bbx, shape: N,8,3 or N,4,2

    gt_range: list
        [xmin, ymin, zmin, xmax, ymax, zmax]
    Returns
    -------
    mask: torch.Tensor
        The mask for bounding box -- True means the
        bbx is within the range and False means the
        bbx is outside the range.
    """

    # mask out the gt bounding box out fixed range (-140, -40, -3, 140, 40 1)
    device = boxes.device
    boundary_lower_range = \
        torch.Tensor(gt_range[:2]).reshape(1, 1, -1).to(device)
    boundary_higher_range = \
        torch.Tensor(gt_range[3:5]).reshape(1, 1, -1).to(device)

    mask = torch.all(
        torch.all(boxes[:, :, :2] >= boundary_lower_range,
                  dim=-1) & \
        torch.all(boxes[:, :, :2] <= boundary_higher_range,
                  dim=-1), dim=-1)

    return mask