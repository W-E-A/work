import torch
import torch.nn as nn
import numpy as np
from mmdet3d.registry import MODELS
from mmengine.model.base_module import BaseModule
from mmdet3d.models.middle_encoders.pillar_scatter import PointPillarsScatter
import torch.nn.functional as F
import math
import os
import os.path as osp

from shapely.geometry import Polygon
from projects.Where2comm.visualization import visualize
import matplotlib.pyplot as plt

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

    regs = reg_result.permute(0, 2, 3, 1).contiguous().reshape(batch_size, -1, 7) # [B, H*W*2, 7]
    boxes = torch.zeros_like(regs) # [B, H*W*2, 7]
    
    anchors = anchor_box.reshape(-1, 7).repeat(batch_size, 1, 1).to(regs.dtype) # [B, H*W*2, 7]
    anchor_d = torch.sqrt(anchors[..., 4] ** 2 + anchors[..., 5] ** 2) # [h*w*2, 1]
    anchor_d = anchor_d.repeat(batch_size, 2, 1).transpose(1, 2) # [B, H*W*2, 2]

    # import pdb
    # pdb.set_trace()

    # Inv-normalize to get xyz
    boxes[..., [0, 1]] = torch.mul(regs[..., [0, 1]], anchor_d) + anchors[..., [0, 1]]
    boxes[..., [2]] = torch.mul(regs[..., [2]], anchors[..., [3]]) + anchors[..., [2]]

    # hwl
    boxes[..., [3, 4, 5]] = torch.exp(regs[..., [3, 4, 5]]) * anchors[..., [3, 4, 5]]
    # yaw angle
    boxes[..., 6] = regs[..., 6] + anchors[..., 6]

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

def postprocess(psm, rm, score_threshold: float, id: int, test_cfg: dict, batch_dict: dict):
    order = test_cfg['order'] # type: ignore
    nms_threshold = test_cfg['nms_threshold'] # type: ignore
    lidar_range = test_cfg['lidar_range'] # type: ignore

    anchor_box = batch_dict['anchor_box'][0].clone() # [H, W, 2, 7] xyzhwly
    cls_pred = torch.sigmoid(psm.permute(0, 2, 3, 1).contiguous()) # [1, 2, H, W] -> [1, H, W, 2]
    cls_pred = cls_pred.reshape(1, -1) # [1, H*W*2]
    box_pred = decode_reg_result(rm, anchor_box) # [1, 14, H, W] [H, W, 2, 7] xyzhwly -> []
    mask = torch.gt(cls_pred, score_threshold) # [1, H*W*2]
    mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)  # [1, H*W*2, 7]
    masked_cls_pred = torch.masked_select(cls_pred[0], mask[0]) # [N, ]
    masked_box_pred = torch.masked_select(box_pred[0], mask_reg[0]).reshape(-1, 7) # [N, 7]

    if len(masked_box_pred) != 0:

        corner_pred_3d = boxes_to_corners_baseline(masked_box_pred, order) # [N, 8, 3]

        # box_pred_2d = corner_to_standup_box(corner_pred_3d) # [N, 4]

        # box_pred_2d_score = torch.cat([box_pred_2d, masked_cls_pred], dim=1) # [N, 5]

        keep_index_1 = remove_large_pred_bbx(corner_pred_3d)
        keep_index_2 = remove_bbx_abnormal_z(corner_pred_3d)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)

        corner_pred_3d_filtered = corner_pred_3d[keep_index] # [n, 8, 3]
        masked_cls_pred_filtered = masked_cls_pred[keep_index] # [n, ]

        keep_index = nms_rotated(corner_pred_3d_filtered, masked_cls_pred_filtered, nms_threshold)
        
        corner_pred_3d_final = corner_pred_3d_filtered[keep_index] # [f, 8, 3]
        masked_cls_pred_final = masked_cls_pred_filtered[keep_index] # [f, ]

        # filter out the prediction out of the range.
        mask = get_mask_for_boxes_within_range(corner_pred_3d_final, lidar_range)
        corner_pred_3d_final = corner_pred_3d_final[mask, :, :]  # [f', 8, 3]
        masked_cls_pred_final = masked_cls_pred_final[mask] # [f', ]

    else:
        corner_pred_3d_final = torch.zeros((0, 8, 3))
        masked_cls_pred_final = torch.zeros((0))

    if id < 0:
        gt_boxes = batch_dict['gt_boxes'][0] # [100, 7]
        gt_mask = batch_dict['gt_mask'][0] # [100, ]
        # gt_object_ids = batch_dict['gt_object_ids'][0] # [N, ]
    else:
        gt_boxes = batch_dict[f'{id}_gt_boxes'][0] # [100, 7]
        gt_mask = batch_dict[f'{id}_gt_mask'][0] # [100, ]
        # gt_object_ids = batch_dict[f'{id}_gt_object_ids'][0] # [N, ]
    masked_gt_boxes = gt_boxes[gt_mask == 1] # [N, 7]
    masked_gt_corner = boxes_to_corners_baseline(masked_gt_boxes, order)
    mask = get_mask_for_boxes_within_range(masked_gt_corner, lidar_range)
    masked_gt_corner_final = masked_gt_corner[mask, :, :]

    return corner_pred_3d_final, masked_cls_pred_final, masked_gt_corner_final


def temp_vis(idx: int,
             id: int,
             vis_dict: dict,
             comm: bool = True,
             save_path: str = './temp_vis',
             save_start: int = 0,
             save_end: int = 10,
             save_wrap: bool = False):
    if (idx >= save_start and idx < save_end) or save_start >= save_end:
        path = f'{save_path}/vis_{idx}'
        os.makedirs(path, exist_ok=True)

        save_path_bev = osp.join(path, f'bev_{id}.png')
        save_path_3d = osp.join(path, f'3d_{id}.png')
        if comm and id >= 0:
            save_path_cmap = osp.join(path, f'cmap_{id}.png')
            save_path_mask = osp.join(path, f'mask_{id}.png')
        save_path_psm = osp.join(path, f'psm_{id}.png')
        save_path_feat = osp.join(path, f'feat_{id}.png')
        save_path_wrap_feat = osp.join(path, f'wrap_feat_{id}.png')
        if comm and id >= 0:
            save_path_wrap_cmap = osp.join(path, f'wrap_cmap_{id}.png')
            save_path_wrap_mask = osp.join(path, f'wrap_mask_{id}.png')

        pc = vis_dict['pc']
        pc_range = vis_dict['pc_range']
        pred_corner = vis_dict['pred_corner']
        gt_corner = vis_dict['gt_corner']
        if comm and id >= 0:
            cmap = vis_dict['cmap']
            mask = vis_dict['mask']
        psm = vis_dict['psm']
        feat = vis_dict['feat']
        if save_wrap:
            rela_pose = vis_dict['rela_pose']
            downsample_rate = vis_dict['downsample_rate']
            voxel_size = vis_dict['voxel_size']
            wrap_feat = warp_affine(
                feat,
                rela_pose,
                downsample_rate,
                voxel_size
            )
            if comm and id >= 0:
                wrap_cmap = warp_affine(
                    cmap, # type: ignore
                    rela_pose,
                    downsample_rate,
                    voxel_size
                )
                wrap_mask = warp_affine(
                    mask, # type: ignore
                    rela_pose,
                    downsample_rate,
                    voxel_size
                )
        visualize(pred_corner, gt_corner, pc, pc_range, save_path_bev, 'bev', vis_gt_box=True, vis_pred_box=True, left_hand=False)
        visualize(pred_corner, gt_corner, pc, pc_range, save_path_3d, '3d', vis_gt_box=True, vis_pred_box=True, left_hand=False)
        if comm and id >= 0:
            target = cmap[0].permute(1, 2, 0).contiguous().squeeze(-1).cpu().numpy() # type: ignore
            target = np.flipud(target)
            plt.imsave(save_path_cmap, target) # type: ignore

            target = mask[0].permute(1, 2, 0).contiguous().squeeze(-1).cpu().numpy() # type: ignore
            target = np.flipud(target)
            plt.imsave(save_path_mask, target) # type: ignore

        target = psm[0].permute(1, 2, 0).contiguous().squeeze(-1) # type: ignore
        target = torch.sigmoid(torch.max(target, dim=-1).values).cpu().numpy()
        target = np.flipud(target)
        plt.imsave(save_path_psm, target) # type: ignore

        target = feat[0].permute(1, 2, 0).contiguous().squeeze(-1) # type: ignore
        target = torch.sigmoid(torch.mean(target,dim=-1)).cpu().numpy()
        target = np.flipud(target)
        plt.imsave(save_path_feat, target) # type: ignore

        if save_wrap:
            target = wrap_feat[0].permute(1, 2, 0).contiguous().squeeze(-1) # type: ignore
            target = torch.sigmoid(torch.mean(target,dim=-1)).cpu().numpy()
            target = np.flipud(target)
            plt.imsave(save_path_wrap_feat, target) # type: ignore

            if comm and id >= 0:
                target = wrap_cmap[0].permute(1, 2, 0).contiguous().squeeze(-1).cpu().numpy() # type: ignore
                target = np.flipud(target)
                plt.imsave(save_path_wrap_cmap, target) # type: ignore

                target = wrap_mask[0].permute(1, 2, 0).contiguous().squeeze(-1).cpu().numpy() # type: ignore
                target = np.flipud(target)
                plt.imsave(save_path_wrap_mask, target) # type: ignore

        
        return True
    elif idx >= save_end:
        return False
    else:
        return True

def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre

def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = det_boxes.cpu().numpy()
        det_score = det_score.cpu().numpy()
        gt_boxes = gt_boxes.cpu().numpy()

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_polygon_list = list(convert_format(det_boxes))
        gt_polygon_list = list(convert_format(gt_boxes))

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)

    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt


def calculate_ap(result_stat, iou):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]

    fp = iou_5['fp']
    tp = iou_5['tp']
    assert len(fp) == len(tp)

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat):
    dump_dict = {}

    ap_30, mrec_30, mpre_30 = calculate_ap(result_stat, 0.30)
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70)

    dump_dict.update({'ap_30': ap_30,
                      'ap_50': ap_50,
                      'ap_70': ap_70,
                      'mpre_50': mpre_50,
                      'mrec_50': mrec_50,
                      'mpre_70': mpre_70,
                      'mrec_70': mrec_70,
                      })

    return ap_30, ap_50, ap_70