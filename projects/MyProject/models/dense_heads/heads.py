# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union, Optional, Sequence, Tuple, Dict
from mmdet3d.registry import MODELS
from mmengine.structures import InstanceData
import torch
from torch import Tensor, nn
import copy
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from mmdet3d.models.utils import clip_sigmoid, draw_heatmap_gaussian, gaussian_radius
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import Det3DDataSample, xywhr2xyxyr
from mmdet3d.models.dense_heads.centerpoint_head import circle_nms, nms_bev
import numpy as np
from ...utils import FeatureWarper
from ..loss_utils import CorrelationLoss


@MODELS.register_module()
class BaseTaskHead(BaseModule):
    def __init__(self,
        task_dict: Dict,
        in_channels: int,
        inter_channels: Optional[int] = None,
        init_cfg = None,
    ):
        super(BaseTaskHead, self).__init__(
            init_cfg=init_cfg
        )

        self.task_heads = nn.ModuleDict()
        inter_channels = in_channels if inter_channels is None else inter_channels
        for task_key, task_dim in task_dict.items():
            self.task_heads[task_key] = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, task_dim, kernel_size=1, padding=0)
            )

        self.fp16_enabled = False

    def loss(self, x):
        pass

    def forward(self, x, targets=None):
        self.float()
        x = x[0]
        return {task_key: task_head(x) for task_key, task_head in self.task_heads.items()}


@MODELS.register_module()
class MTHead(BaseModule):
    def __init__(
            self,
            det_head: Optional[dict] = None,
            motion_head:Optional[dict] = None,
            corr_head:Optional[dict] = None,
            train_cfg: Optional[dict] = None,
            test_cfg: Optional[dict] = None,
            init_cfg: Union[dict, List[dict], None] = None,
            ):
        super().__init__(init_cfg=init_cfg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.det_head = None
        if det_head:
            det_head.update(train_cfg=self.train_cfg)
            det_head.update(test_cfg=self.test_cfg)
            self.det_head = MODELS.build(det_head)
        self.motion_head = None
        if motion_head:
            motion_head.update(train_cfg=self.train_cfg)
            motion_head.update(test_cfg=self.test_cfg)
            self.motion_head = MODELS.build(motion_head)
        self.corr_head = None
        if corr_head:
            corr_head.update(train_cfg=self.train_cfg)
            corr_head.update(test_cfg=self.test_cfg)
            self.corr_head = MODELS.build(corr_head)

    def forward(self,
        feats: Union[Tensor, List[Tensor]],
        det_forward_kwargs: Optional[dict] = None,
        motion_forward_kwargs: Optional[dict] = None,
        corr_forward_kwargs: Optional[dict] = None,):
        return_dict = {}
        # import pdb;pdb.set_trace()
        if not isinstance(feats, Sequence):
            feats = [feats]
        if self.det_head:
            multi_tasks_multi_feats: Tuple[List[Tensor]] = self.det_head(feats, **det_forward_kwargs) # FIXME <100MiB
            return_dict['det_feat'] = multi_tasks_multi_feats # [[t1,],[t2,],[...]]
        if self.motion_head:
            multi_tasks_multi_feats: Tuple[List[Tensor]] = self.motion_head(feats, **motion_forward_kwargs) # FIXME 6000MiB
            return_dict['motion_feat'] = multi_tasks_multi_feats
        if self.corr_head:
            multi_tasks_multi_feats: Tuple[List[Tensor]] = self.corr_head(feats, **corr_forward_kwargs) # FIXME 5000MiB
            return_dict['corr_feat'] = multi_tasks_multi_feats

        return return_dict

    def loss(self,
             feat_dict: dict,
             det_loss_kwargs: Optional[dict] = None,
             motion_loss_kwargs: Optional[dict] = None,
             corr_loss_kwargs: Optional[dict] = None,
             ) -> dict:

        loss_dict = {}
        
        if 'det_feat' in feat_dict:
            multi_tasks_multi_feats = feat_dict['det_feat']
            temp_dict = self.det_head.loss_by_feat(multi_tasks_multi_feats, **det_loss_kwargs) # type: ignore
            loss_dict.update(temp_dict)

        if 'motion_feat' in feat_dict:
            multi_tasks_multi_feats = feat_dict['motion_feat']
            temp_dict = self.motion_head.loss_by_feat(multi_tasks_multi_feats, **motion_loss_kwargs) # type: ignore
            loss_dict.update(temp_dict)

        if 'corr_feat' in feat_dict :
            multi_tasks_multi_feats = feat_dict['corr_feat']
            temp_dict = self.corr_head.loss_by_feat(multi_tasks_multi_feats, **corr_loss_kwargs) # type: ignore
            loss_dict.update(temp_dict)
        
        return loss_dict

    def predict(self,
             feat_dict: dict,
             det_pred_kwargs: Optional[dict] = None,
             motion_pred_kwargs: Optional[dict] = None,
             corr_pred_kwargs: Optional[dict] = None,
             ) -> dict:

        predict_dict = {}
        
        if 'det_feat' in feat_dict:
            multi_tasks_multi_feats = feat_dict['det_feat']
            predict_dict['det_pred'] = self.det_head.predict_by_feat(multi_tasks_multi_feats, **det_pred_kwargs) # type: ignore
            # if return_det_heatmaps:
            #     predict_dict['det_pred'] = self.det_head.predict_heatmaps(multi_tasks_multi_feats) # type: ignore
            # else:
            #     predict_dict['det_pred'] = self.det_head.predict_by_feat(multi_tasks_multi_feats, batch_input_metas) # type: ignore

        if 'motion_feat' in feat_dict:
            multi_tasks_multi_feats = feat_dict['motion_feat']
            predict_dict['motion_pred'] = self.motion_head.predict_by_feat(multi_tasks_multi_feats, **motion_pred_kwargs) # type: ignore
        
        if 'corr_feat' in feat_dict:
            multi_tasks_multi_feats = feat_dict['corr_feat']
            predict_dict['corr_pred'] = self.corr_head.predict_by_feat(multi_tasks_multi_feats, **corr_pred_kwargs) # type: ignore

        return predict_dict


@MODELS.register_module()
class CenterHeadModified(BaseModule):

    def __init__(self,
                 in_channels: Union[List[int], int] = [128],
                 tasks: Optional[List[dict]] = None,
                 bbox_coder: Optional[dict] = None,
                 common_heads: dict = dict(),
                 loss_cls: dict = dict(
                     type='mmdet.GaussianFocalLoss', reduction='mean'),
                 loss_bbox: dict = dict(
                     type='mmdet.L1Loss', reduction='none', loss_weight=0.25),
                 separate_head: dict = dict(
                     type='mmdet.SeparateHead',
                     init_bias=-2.19,
                     final_kernel=3),
                 share_conv_channel: int = 64,
                 num_heatmap_convs: int = 2,
                 conv_cfg: dict = dict(type='Conv2d'),
                 norm_cfg: dict = dict(type='BN2d'),
                 bias: str = 'auto',
                 norm_bbox: bool = True,
                 with_velocity: bool = True,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(CenterHeadModified, self).__init__(init_cfg=init_cfg, **kwargs)

        # FIXME we should rename this variable,
        # for example num_classes_per_task ?
        # {'num_class': 2, 'class_names': ['pedestrian', 'traffic_cone']}]
        # FIXME seems num_classes is useless
        num_classes = [len(t['class_names']) for t in tasks] # type: ignore
        self.class_names = [t['class_names'] for t in tasks] # type: ignore
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_bbox = norm_bbox

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.bbox_coder = TASK_UTILS.build(bbox_coder) # type: ignore
        self.num_anchor_per_locs = [n for n in num_classes]

        # a shared convolution
        self.shared_conv = ConvModule(
            in_channels, # type: ignore
            share_conv_channel,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias)

        self.task_heads = nn.ModuleList()

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))
            separate_head.update(
                in_channels=share_conv_channel, heads=heads, num_cls=num_cls)
            self.task_heads.append(MODELS.build(separate_head))

        #wyc修改
        # self.with_velocity = with_velocity
        if 'vel' in common_heads.keys():
            self.with_velocity = True
        else:
            self.with_velocity = False

        if self.train_cfg:
            self.max_objs = int(self.train_cfg['max_objs'] * self.train_cfg['dense_reg'])
            self.pc_range = self.train_cfg['point_cloud_range']
            self.voxel_size = self.train_cfg['voxel_size']
            self.out_size_factor = self.train_cfg['out_size_factor']

            self.grid_size = torch.tensor([
                np.round((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1]), # H
                np.round((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0]), # W
                np.round((self.pc_range[5] - self.pc_range[2]) / self.voxel_size[2]), # D
            ], dtype=torch.int32) # 1024 1024 1
            self.feature_map_size = [size // self.out_size_factor for size in self.grid_size[:2]] # H W
            self.gaussian_overlap = self.train_cfg['gaussian_overlap']
            self.min_radius = self.train_cfg['min_radius']
            self.code_weights = self.train_cfg.get('code_weights', None)
        if self.test_cfg:
            self.nms_type = self.test_cfg['nms_type']
            self.test_min_radius = self.test_cfg['min_radius']
            self.nms_rescale_factor = self.test_cfg.get(
                'nms_rescale_factor',
                [1.0 for _ in range(len(self.task_heads))])
            self.post_max_size = self.test_cfg['post_max_size']
            self.post_center_limit_range = self.test_cfg['post_center_limit_range']
            self.score_threshold = self.test_cfg['score_threshold']
            self.nms_thr = self.test_cfg['nms_thr']
            self.pre_max_size = self.test_cfg['pre_max_size']
            self.post_max_size = self.test_cfg['post_max_size']

    def forward_single(self, x: Tensor) -> list:
        ret_list = []
        x = self.shared_conv(x)
        for task in self.task_heads:
            ret_list.append(task(x))
        return ret_list

    def forward(self, feats: Union[List[Tensor], Tensor]) -> Tuple[List[Tensor]]:
        """
        [X C H W]
        """
        if not isinstance(feats, Sequence):
            feats = [feats]
        return multi_apply(self.forward_single, feats) # type: ignore

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2) # 8/10
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim) # B M 8/10
        feat = feat.gather(1, ind) # B M 8/10, but only several valid, invalid ind = 0, 0
        if mask is not None:
            # if use mask, only keep the valid
            mask = mask.unsqueeze(2).expand_as(feat) # B M 8/10
            feat = feat[mask] # B, m, 8/10
            feat = feat.view(-1, dim) # B*M 8/10 
        return feat

    def get_targets(
        self,
        batch_gt_instances_3d: List[InstanceData],
    ) -> Tuple[List[Tensor]]:
        """
        task - tensor[b, anyshape] align with prediction format
        """
        heatmaps, anno_boxes, inds, masks = multi_apply(
            self.get_targets_single, batch_gt_instances_3d)
        # Transpose heatmaps
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # Transpose anno_boxes
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # Transpose inds
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        # Transpose inds
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]
        return heatmaps, anno_boxes, inds, masks # type: ignore

    def get_targets_single(self,
                           gt_instances_3d: InstanceData) -> Tuple[Tensor]:
        """Generate training targets for a single sample.

        Args:
            gt_instances_3d (:obj:`InstanceData`): Gt_instances of
                single data sample. It usually includes
                ``bboxes_3d`` and ``labels_3d`` attributes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes
                    are valid.
        """
        gt_labels_3d = gt_instances_3d.labels_3d # type: ignore
        gt_bboxes_3d = gt_instances_3d.bboxes_3d # type: ignore
        track_id = gt_instances_3d.track_id # type: ignore

        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat( # get gravity center box
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
            
        track_id = torch.tensor(track_id, dtype=gt_labels_3d.dtype, device=device)

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        task_ids = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            task_id = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
                task_id.append(track_id[m])
            task_boxes.append(torch.cat(task_box, axis=0).to(device)) # type: ignore
            task_classes.append(torch.cat(task_class).long().to(device))
            task_ids.append(torch.cat(task_id, axis=0).to(device)) # type: ignore
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        task_len = len(self.task_heads)
        for idx in range(task_len):
            heatmap = gt_bboxes_3d.new_zeros((
                len(self.class_names[idx]),
                self.feature_map_size[0].item(),
                self.feature_map_size[1].item()
            )) # type: ignore

            if self.with_velocity:
                anno_box = gt_bboxes_3d.new_zeros((self.max_objs, 10),
                                                dtype=torch.float32)
            else:
                anno_box = gt_bboxes_3d.new_zeros((self.max_objs, 8),
                                                dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((self.max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((self.max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], self.max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1 # sub class

                length = task_boxes[idx][k][3]
                width = task_boxes[idx][k][4]
                length = length / self.voxel_size[0] / self.out_size_factor
                width = width / self.voxel_size[1] / self.out_size_factor

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (width, length),
                        min_overlap=self.gaussian_overlap)
                    radius = max(self.min_radius, int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                        x - self.pc_range[0]
                    ) / self.voxel_size[0] / self.out_size_factor
                    coor_y = (
                        y - self.pc_range[1]
                    ) / self.voxel_size[1] / self.out_size_factor

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < self.feature_map_size[1]
                            and 0 <= center_int[1] < self.feature_map_size[0]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * self.feature_map_size[1] + x <
                            self.feature_map_size[1] * self.feature_map_size[0])

                    ind[new_idx] = y * self.feature_map_size[1] + x
                    mask[new_idx] = 1
                    if self.with_velocity:
                        vx, vy = task_boxes[idx][k][7:9]
                        rot = task_boxes[idx][k][6]
                        box_dim = task_boxes[idx][k][3:6]
                        if self.norm_bbox:
                            box_dim = box_dim.log()
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([x, y], device=device),
                            z.unsqueeze(0), box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0),
                            vx.unsqueeze(0),
                            vy.unsqueeze(0)
                        ])
                    else:
                        rot = task_boxes[idx][k][6]
                        box_dim = task_boxes[idx][k][3:6]
                        if self.norm_bbox:
                            box_dim = box_dim.log()
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([x, y], device=device),
                            z.unsqueeze(0), box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0)
                        ])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks # type: ignore

    def loss_by_feat(self,
                     preds_dicts: Tuple[List[dict]],
                     heatmaps: Optional[List[Tensor]] = None,
                     anno_boxes: Optional[List[Tensor]] = None,
                     inds: Optional[List[Tensor]] = None,
                     masks: Optional[List[Tensor]] = None,
                     ):
        # FIXME single scale only now
        assert isinstance(preds_dicts, Sequence) and isinstance(preds_dicts[0], Sequence) and len(preds_dicts[0]) == 1
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            pred_result = preds_dict[0]
            # heatmap focal loss
            pred_result['heatmap'] = clip_sigmoid(pred_result['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item() # type: ignore
            loss_heatmap = self.loss_cls(
                pred_result['heatmap'], # B c H W
                heatmaps[task_id], # type: ignore  B c H W
                avg_factor=max(num_pos, 1))
            target_box = anno_boxes[task_id] # type: ignore # B M 8/10
            # reconstruct the anno_box from multiple reg heads
            if self.with_velocity:
                pred_result['anno_box'] = torch.cat(
                    (pred_result['reg'], pred_result['height'],
                    pred_result['dim'], pred_result['rot'],
                    pred_result['vel']),
                    dim=1) # B 10 H W
            else:
                pred_result['anno_box'] = torch.cat(
                (pred_result['reg'], pred_result['height'],
                 pred_result['dim'], pred_result['rot']),
                dim=1) # B 8 H W

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id] # type: ignore B M
            num = masks[task_id].float().sum() # type: ignore B M
            pred = pred_result['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3)) # B H*W 8/10
            pred = self._gather_feat(pred, ind) # B M 8/10
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float() # type: ignore # B M 8/10
            isnotnan = (~torch.isnan(target_box)).float() # B M 8/10
            mask *= isnotnan # B M 8/10
            bbox_weights = mask * mask.new_tensor(self.code_weights) # B M 8/10 * 8/10
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4)) # invaid will be masked by the weight
            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
        return loss_dict

    def predict_heatmaps(self, preds_dicts: Tuple[List[dict]],
                         ) -> List[Tensor]:
        heatmaps = []
        # FIXME single scale only now
        assert isinstance(preds_dicts, Sequence) and isinstance(preds_dicts[0], Sequence) and len(preds_dicts[0]) == 1
        for task_id, preds_dict in enumerate(preds_dicts):
            pred_result = preds_dict[0]
            heatmaps.append(pred_result['heatmap'].sigmoid()) # B c H W
        return heatmaps

    def predict_by_feat(self, preds_dicts: Tuple[List[dict]],
                        batch_input_metas: List[dict],
                        ) -> List[InstanceData]:
        rets = []
        # FIXME single scale only now
        assert isinstance(preds_dicts, Sequence) and isinstance(preds_dicts[0], Sequence) and len(preds_dicts[0]) == 1
        batch_size = len(batch_input_metas)
        for task_id, preds_dict in enumerate(preds_dicts):
            pred_result = preds_dict[0]
            num_class_with_bg = self.num_classes[task_id] # c
            batch_heatmap = pred_result['heatmap'].sigmoid() # B c H W

            batch_reg = pred_result['reg'] # B 2 H W
            batch_hei = pred_result['height'] # B 1 H W

            if self.norm_bbox:
                batch_dim = torch.exp(pred_result['dim']) # B 3 H W
            else:
                batch_dim = pred_result['dim']

            batch_rots = pred_result['rot'][:, 0].unsqueeze(1) # B 1 H W
            batch_rotc = pred_result['rot'][:, 1].unsqueeze(1) # B 1 H W

            if self.with_velocity and 'vel' in pred_result:
                batch_vel = pred_result['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode( # FIXME watch this
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id,)
            assert self.nms_type in ['circle', 'rotate']
            batch_reg_preds = [box['bboxes'] for box in temp] # B * box
            batch_cls_preds = [box['scores'] for box in temp] # B * score
            batch_cls_labels = [box['labels'] for box in temp] # B * score
            if self.nms_type == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_min_radius[task_id],
                            post_max_size=self.post_max_size),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds,
                                             batch_reg_preds,
                                             batch_cls_labels,
                                             batch_input_metas,
                                             task_id)) # task * [ batch * [dict, ...]]

        ret_list = []
        for i in range(batch_size):
            temp_instances = InstanceData()
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets]) # center box
                    # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = batch_input_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size, (0.5, 0.5, 0.5))
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class # exp. ('a', 'b'), ('c'), ('d', 'e') task -> 0,1 2 3,4
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            temp_instances.bboxes_3d = bboxes # type: ignore all task
            temp_instances.scores_3d = scores # type: ignore
            temp_instances.labels_3d = labels # type: ignore
            ret_list.append(temp_instances)
        return ret_list

    def get_task_detections(self, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels, img_metas,
                            task_id):
        predictions_dicts = []
        post_center_range = self.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device)

        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):

            nms_rescale_factor = self.nms_rescale_factor[task_id]
            if isinstance(nms_rescale_factor, list):
                for cid in range(len(nms_rescale_factor)):
                    box_preds[cls_labels == cid, 3:6] = \
                    box_preds[cls_labels == cid, 3:6] * nms_rescale_factor[cid]
            else:
                box_preds[:, 3:6] = box_preds[:, 3:6] * nms_rescale_factor

            # Apply NMS in bird eye view

            # get the highest score per prediction, then apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1) # N, 1 -> N
                top_labels = torch.zeros( # onehot
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long) # [1, 1, 1, ...] -> [0, 0, 0]

            else:
                top_labels = cls_labels.long() # onehot
                top_scores = cls_preds.squeeze(-1) # N, 1 -> N

            if self.score_threshold > 0.0: # thres_select
                thresh = torch.tensor(
                    [self.score_threshold],
                    device=cls_preds.device).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0: # thres_select
                if self.score_threshold > 0.0:
                    box_preds = box_preds[top_scores_keep] # type: ignore # N, 7/9
                    top_labels = top_labels[top_scores_keep] # type: ignore # N, c

                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](
                    box_preds[:, :], self.bbox_coder.code_size).bev)
                # the nms in 3d detection just remove overlap boxes.

                if isinstance(self.nms_thr, list):
                    nms_thresh = self.nms_thr[task_id]
                else:
                    nms_thresh = self.nms_thr

                selected = nms_bev(
                    boxes_for_nms, # type: ignore
                    top_scores,
                    thresh=nms_thresh,
                    pre_max_size=self.pre_max_size,
                    post_max_size=self.post_max_size)
            else:
                selected = [] # all score less than score_threshold

            if isinstance(nms_rescale_factor, list):
                for cid in range(len(nms_rescale_factor)):
                    box_preds[cls_labels == cid, 3:6] = \
                    box_preds[cls_labels == cid, 3:6] / nms_rescale_factor[cid]
            else:
                box_preds[:, 3:6] = box_preds[:, 3:6] / nms_rescale_factor

            # if selected is not None:
            selected_boxes = box_preds[selected] # n, 7/9
            selected_labels = top_labels[selected] # n, c
            selected_scores = top_scores[selected] # n

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = dict(
                        bboxes=final_box_preds[mask], # m 7/9
                        scores=final_scores[mask], # m
                        labels=final_labels[mask]) # m
                else:
                    predictions_dict = dict(
                        bboxes=final_box_preds,
                        scores=final_scores,
                        labels=final_labels)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size],
                                       dtype=dtype,
                                       device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device))

            predictions_dicts.append(predictions_dict)
        return predictions_dicts


@MODELS.register_module()
class CorrGenerateHead(BaseModule):
    def __init__(
        self,
        pc_range,
        voxel_size,
        n_future_and_present: int = 0,
        label_size: int = 6,
        in_channels: Union[List[int], int] = [128],
        loss_cfg: Optional[dict] = dict(
            type='CorrelationLoss',
            gamma = 2.0,
            smooth_beta = 0.5,
            pos_weight = 1.0,
            neg_weight = 1.0,
        ),
        separate_head: dict = dict(
            type='mmdet.SeparateHead',
            init_bias=-2.19,
            final_kernel=3),
        share_conv_channel: int = 64,
        num_heatmap_convs: int = 2,
        conv_cfg: dict = dict(type='Conv2d'),
        norm_cfg: dict = dict(type='BN2d'),
        bias: str = 'auto',
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        **kwargs
        ):
        super(CorrGenerateHead, self).__init__(init_cfg=init_cfg, **kwargs)

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.voxel_size = np.array(self.voxel_size).astype(np.float32)
        self.grid_size = np.array([
            np.round((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1]), # H
            np.round((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0]), # W
            np.round((self.pc_range[5] - self.pc_range[2]) / self.voxel_size[2]), # D
        ]).astype(np.int32)
        self.warper = FeatureWarper(self.pc_range)
        self.add_channels = n_future_and_present * label_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.heatmap_criterion = MODELS.build(loss_cfg)

        # a shared convolution
        self.shared_conv = ConvModule(
            in_channels + self.add_channels, # type: ignore
            share_conv_channel,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias)

        heads = dict(heatmap=(1, num_heatmap_convs))
        separate_head.update(
            in_channels=share_conv_channel, heads=heads, num_cls=1)
        self.corr_head = MODELS.build(separate_head)

        if self.train_cfg:
            self.max_objs = int(self.train_cfg['corr_max_objs'] * self.train_cfg['corr_dense_reg'])
            self.gaussian_overlap = self.train_cfg['corr_gaussian_overlap']
            self.min_radius = self.train_cfg['corr_min_radius']

    def forward(self, feats: Union[List[Tensor], Tensor], ego_motion_inputs=None):
        all_ret_list = []
        if not ego_motion_inputs:
            all_ret_list.append([self.corr_head(self.shared_conv(feats[0]))])
            return all_ret_list
        elif not isinstance(ego_motion_inputs, Sequence):
            ego_motion_inputs = [ego_motion_inputs]
        for input in ego_motion_inputs:
            b, _, _, h, w = input.shape
            input = input.view(b, -1, h, w)
            all_ret_list.append([self.corr_head(self.shared_conv(torch.cat([feats[0], input], dim=1)))])
        return all_ret_list

    def loss_by_feat(self,
                     preds_list,
                     heatmaps = None,
                     gt_masks = None,
                     dilate_heatmaps = None,
                     loss_names = None,
                     gt_thres: float = 0.3):
        assert heatmaps is not None and gt_masks is not None and dilate_heatmaps is not None
        loss_dict = {}
        names = list(range(len(preds_list)))
        if loss_names:
            names = loss_names

        # import pdb;pdb.set_trace()

        # for name, pred_result, heatmap, gt_mask, dilate_heatmap in zip(names, preds_list, heatmaps, gt_masks, dilate_heatmaps):
        #     pred_result = pred_result[0]
        #     pred_heatmap = clip_sigmoid(pred_result['heatmap']) # b, 1, h, w
        #     loss_heatmap = self.heatmap_criterion(
        #         pred_heatmap, # b, 1, h, w
        #         heatmap, # b, 1, h, w
        #         gt_mask, # b, 1, h, w
        #         dilate_heatmap, # b, 1, h, w
        #     )
        #     loss_dict[f'{name}.loss_corr_heatmap'] = loss_heatmap

        pred_heatmap = torch.stack([clip_sigmoid(v[0]['heatmap']) for v in preds_list], dim=0)
        loss_heatmap = self.heatmap_criterion(
            pred_heatmap, # c-1, b, 1, h, w
            heatmaps, # c-1, b, 1, h, w
            gt_masks, # c-1, b, 1, h, w
            dilate_heatmaps, # c-1, b, 1, h, w
        )
        loss_dict[f'loss_corr_heatmap'] = loss_heatmap

        return loss_dict

    def predict_by_feat(self, preds_dicts: Tuple[List[dict]]) -> List[Tensor]:
        heatmaps = []
        assert isinstance(preds_dicts, Sequence) and isinstance(preds_dicts[0], Sequence) and len(preds_dicts[0]) == 1
        for task_id, preds_dict in enumerate(preds_dicts):
            pred_result = preds_dict[0]
            heatmaps.append(pred_result['heatmap'].sigmoid()) # B c H W
        return heatmaps

    def prepare_ego_labels(self, batch_list, label_sampling_mode='nearest', center_sampling_mode='nearest'):
        if not isinstance(batch_list, Sequence):
            batch_list = [batch_list]
        return_list = []
        for batch in batch_list:
            labels = {}
            ego_motion_inputs = []
            segmentation_labels = torch.stack(batch['motion_segmentation']) # B, len, H, W
            instance_center_labels = torch.stack(batch['instance_centerness']) # B, len, H, W
            instance_offset_labels = torch.stack(batch['instance_offset']) # B, len, 2, H, W
            instance_flow_labels = torch.stack(batch['instance_flow']) # B, len, 2, H, W
            gt_instance = torch.stack(batch['motion_instance']) # B, len, H, W
            future_egomotion = torch.stack(batch['future_egomotion']) # B, len, 4, 4
            bev_transform = batch.get('aug_transform', None)

            if bev_transform is not None:
                bev_transform = bev_transform.float()

            segmentation_labels = self.warper.cumulative_warp_features_reverse(
                segmentation_labels.float().unsqueeze(2), # B, len, 1, H, W
                future_egomotion,
                mode=label_sampling_mode, bev_transform=bev_transform,
            ).long().contiguous() # to long(0 or 1)
            labels['segmentation'] = segmentation_labels  # B, len, 1, H, W
            ego_motion_inputs.append(segmentation_labels)

            # Warp instance labels to present's reference frame
            gt_instance = self.warper.cumulative_warp_features_reverse(
                gt_instance.float().unsqueeze(2), # B, len, 1, H, W
                future_egomotion,
                mode=label_sampling_mode, bev_transform=bev_transform,
            ).long().contiguous()[:, :, 0] # to long(0, 1~254, 255) # B, len, H, W
            labels['instance'] = gt_instance

            instance_center_labels = self.warper.cumulative_warp_features_reverse(
                instance_center_labels,
                future_egomotion,
                mode=center_sampling_mode, bev_transform=bev_transform,
            ).contiguous()
            labels['centerness'] = instance_center_labels # B, len, 1, H, W

            instance_offset_labels = self.warper.cumulative_warp_features_reverse(
                instance_offset_labels,
                future_egomotion,
                mode=label_sampling_mode, bev_transform=bev_transform,
            ).contiguous()
            labels['offset'] = instance_offset_labels # B, len, 2, H, W

            ego_motion_inputs.append(instance_center_labels)
            ego_motion_inputs.append(instance_offset_labels)

            instance_flow_labels = self.warper.cumulative_warp_features_reverse(
                instance_flow_labels,
                future_egomotion,
                mode=label_sampling_mode, bev_transform=bev_transform,
            ).contiguous()
            labels['flow'] = instance_flow_labels # B, len, 2, H, W
            ego_motion_inputs.append(instance_flow_labels)

            if len(ego_motion_inputs) > 0:
                ego_motion_inputs = torch.cat(
                    ego_motion_inputs, dim=2) #  # B, len, 1+1+2+2, H, W
            
            return_list.append((labels, ego_motion_inputs)) # dict tensor B len 1+1+2+2 256, 256

        return_list = tuple(map(list, zip(*return_list)))
        return return_list

    def prepare_corr_heatmaps(self, heatmaps, **kwargs):
        # intput list of c-1 h w tensor
        heatmaps = torch.stack(heatmaps, dim=0).unsqueeze(2).permute(1, 0, 2, 3, 4).contiguous() # c-1, B, 1, h, w
        ret_list = [heatmaps]
        for _, v in kwargs.items():
            ret_list.append(torch.stack(v, dim=0).unsqueeze(2).permute(1, 0, 2, 3, 4).contiguous()) # c-1, B, 1, h, w
        return tuple(ret_list)

    def get_corr_heatmaps(
        self,
        batch_instances: List[InstanceData],
    ) -> Tuple[List[Tensor]]:
        heatmaps = multi_apply(self.get_corr_single, batch_instances)
        heatmaps = [torch.stack(corr_) for corr_ in heatmaps]
        return heatmaps # [task1, task2, task3, ...]

    def get_corr_single(self, gt_instances_3d):
        assert 'correlation' in gt_instances_3d
        correlation = gt_instances_3d.correlation
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        device = gt_bboxes_3d.device
        gt_bboxes_3d = torch.cat( # get gravity center box
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
        
        draw_gaussian = draw_heatmap_gaussian
        heatmaps = []
        task_len = correlation.shape[-1]
        num_objs = min(gt_bboxes_3d.shape[0], self.max_objs)
        for idx in range(task_len):
            heatmap = gt_bboxes_3d.new_zeros((
                1,
                self.grid_size[0],
                self.grid_size[1]
            )) # type: ignore
            corr = correlation[:, idx]
            for k in range(num_objs):
                score = corr[k]
                length = gt_bboxes_3d[k][3] / self.voxel_size[0]
                width = gt_bboxes_3d[k][4] / self.voxel_size[1]

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (width, length),
                        min_overlap=self.gaussian_overlap)
                    radius = max(self.min_radius, int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = gt_bboxes_3d[k][0], gt_bboxes_3d[k][1], gt_bboxes_3d[k][2]

                    coor_x = (
                        x - self.pc_range[0]
                    ) / self.voxel_size[0]
                    coor_y = (
                        y - self.pc_range[1]
                    ) / self.voxel_size[1]

                    center = torch.tensor([coor_x, coor_y],
                                        dtype=torch.float32,
                                        device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < self.grid_size[1]
                            and 0 <= center_int[1] < self.grid_size[0]):
                        continue

                    draw_gaussian(heatmap[0], center_int, radius, k=score)

                    x, y = center_int[0], center_int[1]
                    assert (y * self.grid_size[1] + x < self.grid_size[1] * self.grid_size[0])
            heatmaps.append(heatmap)
        return heatmaps




