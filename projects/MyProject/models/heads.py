# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union, Optional, Sequence, Tuple, Dict
from mmdet.models.utils import multi_apply
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
import torch
from torch import Tensor, nn
import copy
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from mmdet3d.models.utils import (clip_sigmoid, draw_heatmap_gaussian,
                                  gaussian_radius)
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import Det3DDataSample, xywhr2xyxyr
from mmdet3d.models.dense_heads.centerpoint_head import circle_nms, nms_bev
import math

# @MODELS.register_module()
# class MotionHead(BaseModule):
#     def __init__(self,
#                  in_channels: int = 128,
#                  inner_channels: Optional[int] = None,
#                  history_length: int = 3,
#                  future_length: int = 1,
#                  future_discount: float = 0.95,
#                  future_dim: int = 6,
#                  common_heads: Optional[List[dict]] = None,
#                  num_gru_blocks: int = 3,
#                  prob_enable: bool = True,
#                  prob_on_foreground: bool = False,
#                  prob_spatially: bool = False,
#                  prob_each_future: bool = False,
#                  prob_latent_dim: int = 32,
#                  distribution_log_sigmas: tuple = (-5.0, 5.0),
#                  detach_state=True,
#                  norm_cfg: dict = dict(type='BN2d'),
#                  init_bias: float = -2.19,
#                  train_cfg: Optional[dict] = None,
#                  test_cfg: Optional[dict] = None,
#                  init_cfg: Union[dict, List[dict], None] = None,

#                  class_weights=None,
#                  use_topk=True,
#                  topk_ratio=0.25,
#                  ignore_index=255,
#                  posterior_with_label=False,
#                  sample_ignore_mode='all_valid',
#                  using_focal_loss=False,
#                  focal_cfg=dict(type='mmdet.GaussianFocalLoss', reduction='none'),
#                  loss_weights=None,
#                  ):
#         super(MotionHead, self).__init__(init_cfg=init_cfg)
#         # FIXME

#         self.in_channels = in_channels
#         self.inner_channels = inner_channels
#         self.history_length = history_length
#         self.future_length = future_length
#         self.future_discount = future_discount
#         self.future_dim = future_dim
#         self.train_cfg = train_cfg
#         self.test_cfg = test_cfg
    
#     def forward_single(self, x: Tensor, ) -> List[dict]:
#         N, C, H, W = x.shape
#         if self.future_length > 0:
#             pass
#         else:
#             pass

#     def forward(self, feats: Union[Tensor, List[Tensor]], ) -> Tuple[List[Tensor]]:
#         if not isinstance(feats, Sequence):
#             feats = [feats]
#         return multi_apply(self.forward_single, feats, ) # type: ignore


@MODELS.register_module()
class MTHead(BaseModule):
    def __init__(
            self,
            det_head: Optional[dict] = None,
            motion_head:Optional[dict] = None,
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

    def forward(self, feats: Union[Tensor, List[Tensor]]) -> dict:
        return_dict = {}
        if not isinstance(feats, Sequence):
            feats = [feats]
        if self.det_head:
            multi_tasks_multi_feats: Tuple[List[Tensor]] = self.det_head(feats)
            return_dict['det_feat'] = multi_tasks_multi_feats # [[t1,],[t2,],[...]]
        if self.motion_head:
            multi_tasks_multi_feats: Tuple[List[Tensor]] = self.motion_head(feats)
            return_dict['motion_feat'] = multi_tasks_multi_feats

        return return_dict

    def loss(self,
             feat_dict: dict,
             batch_det_instances: Optional[List[InstanceData]] = None,
             batch_motion_instances: Optional[List[InstanceData]] = None,
             gather_task_loss: bool = False
             ) -> dict:

        loss_dict = {}
        
        if 'det_feat' in feat_dict and batch_det_instances != None:
            multi_tasks_multi_feats = feat_dict['det_feat']
            # TODO just one scale here
            temp_dict = self.det_head.loss_by_feat(multi_tasks_multi_feats, batch_det_instances)
            if gather_task_loss:
                gather_dict = {}
                for key in temp_dict.keys():
                    if 'loss' in key:
                        cut_key = key.split('.')[-1]
                        if cut_key not in gather_dict:
                            gather_dict[cut_key] = []
                        gather_dict[cut_key].append(temp_dict[key])
                loss_dict.update(gather_dict)
            else:
                loss_dict.update(temp_dict)

        if 'motion_feat' in feat_dict and batch_motion_instances != None:
            multi_tasks_multi_feats = feat_dict['motion_feat']
            # TODO just one scale here
            temp_dict = self.motion_head.loss_by_feat(multi_tasks_multi_feats, batch_motion_instances)
            if gather_task_loss:
                gather_dict = {}
                for key in temp_dict.keys():
                    if 'loss' in key:
                        cut_key = key.split('.')[-1]
                        if cut_key not in gather_dict:
                            gather_dict[cut_key] = []
                        gather_dict[cut_key].append(temp_dict[key])
                loss_dict.update(gather_dict)
            else:
                loss_dict.update(temp_dict)
        
        return loss_dict

    def predict(self,
             feat_dict: dict,
             batch_input_metas: Optional[List[dict]] = None,
             ) -> dict:

        predict_dict = {}
        
        if 'det_feat' in feat_dict and batch_input_metas != None:
            multi_tasks_multi_feats = feat_dict['det_feat']
            # TODO just one scale here
            batch_result_instances = self.det_head.predict_by_feat(multi_tasks_multi_feats, batch_input_metas)
            predict_dict['det_pred'] = batch_result_instances

        # if 'motion_feat' in feat_dict and batch_input_metas != None:
        #     multi_tasks_multi_feats = feat_dict['motion_feat']
        #     # TODO just one scale here
        #     batch_result_instances = self.motion_head.predict_by_feat(multi_tasks_multi_feats, batch_input_metas)
        #     predict_dict['motion_pred'] = batch_result_instances

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
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 with_velocity: bool = True,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(CenterHeadModified, self).__init__(init_cfg=init_cfg, **kwargs)

        # TODO we should rename this variable,
        # for example num_classes_per_task ?
        # {'num_class': 2, 'class_names': ['pedestrian', 'traffic_cone']}]
        # TODO seems num_classes is useless
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

        self.with_velocity = with_velocity
        if self.train_cfg:
            self.max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
            self.pc_range = self.train_cfg['point_cloud_range']
            self.voxel_size = self.train_cfg['voxel_size']
            self.out_size_factor = self.train_cfg['out_size_factor']

            self.grid_size = [
                math.ceil((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1]),
                math.ceil((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0]),
                math.ceil((self.pc_range[5] - self.pc_range[2]) / self.voxel_size[2]),
            ]
            self.feature_map_size = [size // self.out_size_factor for size in self.grid_size[:2]]
            self.gaussian_overlap = self.train_cfg['gaussian_overlap']
            self.min_radius = self.train_cfg['min_radius']
            self.code_weights = self.train_cfg.get('code_weights', None)
        if self.test_cfg:
            self.nms_type = self.test_cfg['nms_type']
            self.test_min_radius = self.test_cfg['min_radius']
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
        if not isinstance(feats, Sequence):
            feats = [feats]
        return multi_apply(self.forward_single, feats) # type: ignore

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor, optional): Mask of the feature map with the
                shape of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def get_targets(
        self,
        batch_gt_instances_3d: List[InstanceData],
    ) -> Tuple[List[Tensor]]:
        """Generate targets.

        How each output is transformed:

            Each nested list is transposed so that all same-index elements in
            each sub-list (1, ..., N) become the new sub-lists.
                [ [a0, a1, a2, ... ], [b0, b1, b2, ... ], ... ]
                ==> [ [a0, b0, ... ], [a1, b1, ... ], [a2, b2, ... ] ]

            The new transposed nested list is converted into a list of N
            tensors generated by concatenating tensors in the new sub-lists.
                [ tensor0, tensor1, tensor2, ... ]

        Args:
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and\
                ``labels_3d`` attributes.

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including
                    the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the
                    position of the valid boxes.
                - list[torch.Tensor]: Masks indicating which
                    boxes are valid.
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
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)

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
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device)) # type: ignore
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), self.feature_map_size[1].item(),
                 self.feature_map_size[0].item())) # type: ignore

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
                cls_id = task_classes[idx][k] - 1

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
                    if not (0 <= center_int[0] < self.feature_map_size[0]
                            and 0 <= center_int[1] < self.feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * self.feature_map_size[0] + x <
                            self.feature_map_size[0] * self.feature_map_size[1])

                    ind[new_idx] = y * self.feature_map_size[0] + x
                    mask[new_idx] = 1
                    if self.with_velocity:
                        # TODO: support other outdoor dataset
                        vx, vy = task_boxes[idx][k][7:]
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
                        # TODO: support other outdoor dataset
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

    def loss(self, pts_feats: List[Tensor],
             batch_data_samples: List[Det3DDataSample], *args,
             **kwargs) -> Dict[str, Tensor]:
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict: Losses of each branch.
        """
        outs = self(pts_feats)
        batch_gt_instance_3d = []
        for data_sample in batch_data_samples:
            batch_gt_instance_3d.append(data_sample.gt_instances_3d)
        losses = self.loss_by_feat(outs, batch_gt_instance_3d)
        return losses

    def loss_by_feat(self, preds_dicts: Tuple[List[dict]],
                     batch_gt_instances_3d: List[InstanceData], *args,
                     **kwargs):
        """Loss function for CenterHead.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results of
                multiple tasks. The outer tuple indicate  different
                tasks head, and the internal list indicate different
                FPN level.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and\
                ``labels_3d`` attributes.

        Returns:
            dict[str,torch.Tensor]: Loss of heatmap and bbox of each task.
        """

        heatmaps, anno_boxes, inds, masks = self.get_targets( # type: ignore
            batch_gt_instances_3d)
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            if self.with_velocity:
                preds_dict[0]['anno_box'] = torch.cat(
                    (preds_dict[0]['reg'], preds_dict[0]['height'],
                    preds_dict[0]['dim'], preds_dict[0]['rot'],
                    preds_dict[0]['vel']),
                    dim=1)
            else:
                preds_dict[0]['anno_box'] = torch.cat(
                (preds_dict[0]['reg'], preds_dict[0]['height'],
                 preds_dict[0]['dim'], preds_dict[0]['rot']),
                dim=1)

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            
            bbox_weights = mask * mask.new_tensor(self.code_weights)
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
        return loss_dict

    def predict(self,
                pts_feats: Dict[str, torch.Tensor],
                batch_data_samples: List[Det3DDataSample],
                rescale=True,
                **kwargs) -> List[InstanceData]:
        """
        Args:
            pts_feats (dict): Point features..
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes meta information of data.
            rescale (bool): Whether rescale the resutls to
                the original scale.

        Returns:
            list[:obj:`InstanceData`]: List of processed predictions. Each
            InstanceData contains 3d Bounding boxes and corresponding
            scores and labels.
        """
        preds_dict = self(pts_feats)
        batch_size = len(batch_data_samples)
        batch_input_metas = []
        for batch_index in range(batch_size):
            metainfo = batch_data_samples[batch_index].metainfo
            batch_input_metas.append(metainfo)

        results_list = self.predict_by_feat(
            preds_dict, batch_input_metas, rescale=rescale, **kwargs)
        return results_list

    def predict_by_feat(self, preds_dicts: Tuple[List[dict]],
                        batch_input_metas: List[dict], *args,
                        **kwargs) -> List[InstanceData]:
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results of
                multiple tasks. The outer tuple indicate  different
                tasks head, and the internal list indicate different
                FPN level.
            batch_input_metas (list[dict]): Meta info of multiple
                inputs.

        Returns:
            list[:obj:`InstanceData`]: Instance prediction
            results of each sample after the post process.
            Each item usually contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes_3d (:obj:`LiDARInstance3DBoxes`): Prediction
                  of bboxes, contains a tensor with shape
                  (num_instances, 7) or (num_instances, 9), and
                  the last 2 dimensions of 9 is
                  velocity.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)
            assert self.nms_type in ['circle', 'rotate']
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
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
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels,
                                             batch_input_metas))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            temp_instances = InstanceData()
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = batch_input_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            temp_instances.bboxes_3d = bboxes # type: ignore
            temp_instances.scores_3d = scores # type: ignore
            temp_instances.labels_3d = labels # type: ignore
            ret_list.append(temp_instances)
        return ret_list

    def get_task_detections(self, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels, img_metas):
        """Rotate nms for each task.

        Args:
            num_class_with_bg (int): Number of classes for the current task.
            batch_cls_preds (list[torch.Tensor]): Prediction score with the
                shape of [N].
            batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].
            batch_cls_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:

                -bboxes (torch.Tensor): Prediction bboxes after nms with the
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the
                    shape of [N].
        """
        predictions_dicts = []
        post_center_range = self.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device)

        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):

            # Apply NMS in bird eye view

            # get the highest score per prediction, then apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long)

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            if self.score_threshold > 0.0:
                thresh = torch.tensor(
                    [self.score_threshold],
                    device=cls_preds.device).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if self.score_threshold > 0.0:
                    box_preds = box_preds[top_scores_keep] # type: ignore
                    top_labels = top_labels[top_scores_keep] # type: ignore

                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](
                    box_preds[:, :], self.bbox_coder.code_size).bev)
                # the nms in 3d detection just remove overlap boxes.

                selected = nms_bev(
                    boxes_for_nms, # type: ignore
                    top_scores,
                    thresh=self.nms_thr,
                    pre_max_size=self.pre_max_size,
                    post_max_size=self.post_max_size)
            else:
                selected = []

            # if selected is not None:
            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

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
                        bboxes=final_box_preds[mask],
                        scores=final_scores[mask],
                        labels=final_labels[mask])
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