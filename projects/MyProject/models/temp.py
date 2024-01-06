if multi_feat: # type: ignore
    # [bs, 128*3, 256, 256] or 3 * [N, C, H, W]
    if mode == 'loss':
        heatmaps, anno_boxes, inds, masks = self.base_model.pts_bbox_head.get_targets(batch_single_instances) # type: ignore
        # for sample in batch_single_samples
        # task_mask -> [ [], [], [] ]
        # task_box [ [task1box, n], [task2box, m], ...]
        # task_classes [ [task1cls, ], [task2cls, ], ... ] 0 means background so begin with 1
    multi_task_pred = self.base_model.pts_bbox_head(multi_feat) # TODO # type: ignore [[t1,t2,t3,...], [], []] multi feats - multi tasks(task dict multi result)
    # TODO multi feat use diff out_size_factor
    # task means class seperated task
    task_size = len(multi_task_pred)
    for task_id, multi_pred in enumerate(multi_task_pred):
        task_pred = multi_pred[0]
        if mode == 'loss':
            loss_dict = {}
            heatmap = clip_sigmoid(task_pred['heatmap']) # [N, 1, H, W] ? 
            pred_bbox = torch.cat(
                        (task_pred['reg'], task_pred['height'],
                        task_pred['dim'], task_pred['rot'],
                        task_pred['vel']),
                        dim=1) # [N, 2+1+3+2+2, H, W]
            gt_heatmap = heatmaps[task_id] # type: ignore [N, 1, H, W]
            num_pos = gt_heatmap.eq(1).float().sum().item()
            gt_anno_box = anno_boxes[task_id] # type: ignore [N, max, 10]
            gt_ind = inds[task_id] # type: ignore [N, max, ]
            gt_mask = masks[task_id] # type: ignore [N, max, ]
            num_gt = gt_mask.float().sum()
            pred_bbox = pred_bbox.permute(0, 2, 3, 1).contiguous()
            pred_bbox = pred_bbox.view(batch_size, -1, pred_bbox.shape[-1]) # [N, H*W, 10]
            # [N, H*W, 10] [N, max, 10]
            pred_bbox = self.base_model.pts_bbox_head._gather_feat(pred_bbox, gt_ind) # type: ignore
            gt_mask = gt_mask.unsqueeze(2).expand_as(gt_anno_box).float()
            isnotnan = (~torch.isnan(gt_anno_box)).float()
            gt_mask *= isnotnan

            code_weights = self.base_model.pts_bbox_head.train_cfg.get('code_weights', None) # type: ignore
            bbox_weights = gt_mask * gt_mask.new_tensor(code_weights)

            loss_heatmap = self.base_model.pts_bbox_head.loss_cls( # TODO # type: ignore
                                            heatmap,
                                            gt_heatmap,
                                            avg_factor=max(num_pos, 1))
            loss_bbox = self.base_model.pts_bbox_head.loss_bbox( # TODO # type: ignore
                                pred_bbox, gt_anno_box, bbox_weights, avg_factor=(num_gt + 1e-4))

            loss_dict['HMloss'] = loss_heatmap # type: ignore
            loss_dict['BXloss'] = loss_bbox # type: ignore
            scene_loss[i][j].append(loss_dict) # type: ignore
        elif mode == 'predict':
            num_class_with_bg = self.base_model.pts_bbox_head.num_classes[task_id] # type: ignore
            batch_heatmap = task_pred['heatmap'].sigmoid() # [N, 1, H, W]
            batch_reg = task_pred['reg'] # [N, 2, H, W]
            batch_height = task_pred['height'] # [N, 1, H, W]

            if self.base_model.pts_bbox_head.train_cfg.get('norm_bbox', False): # type: ignore
                batch_dim = torch.exp(task_pred['dim']) # [N, 3, H, W]
            else:
                batch_dim = task_pred['dim']

            batch_rots = task_pred['rot'][:, 0].unsqueeze(1) # [N, 1, H, W]
            batch_rotc = task_pred['rot'][:, 1].unsqueeze(1) # [N, 2, H, W]

            if 'vel' in task_pred:
                batch_vel = task_pred['vel'] # [N, 2, H, W]
            else:
                batch_vel = None

            temp = self.base_model.pts_bbox_head.bbox_coder.decode( # type: ignore
                        batch_heatmap,
                        batch_rots,
                        batch_rotc,
                        batch_height,
                        batch_dim,
                        batch_vel,
                        reg=batch_reg,
                        task_id=task_id)
            assert self.test_cfg['nms_type'] in ['circle', 'rotate'] # type: ignore
            batch_reg_preds = [box['bboxes'] for box in temp] # batch size list of N, 9
            batch_cls_preds = [box['scores'] for box in temp] # batch size list of N,
            batch_cls_labels = [box['labels'] for box in temp] # batch size list of N,

            if self.test_cfg['nms_type'] == 'circle': # type: ignore
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
                            self.test_cfg['min_radius'][task_id], # type: ignore
                            post_max_size=self.test_cfg['post_max_size']), # type: ignore
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                scene_ret[i][j].append(ret_task) # type: ignore
            else:
                scene_ret[i][j].append( # type: ignore
                    self.base_model.pts_bbox_head.get_task_detections( # type: ignore
                        num_class_with_bg,
                        batch_cls_preds,
                        batch_reg_preds,
                        batch_cls_labels,
                        batch_single_input_metas) # batch list of predictions_dict
                    )
    if mode == 'predict':
        ret_list = []
        for b in range(batch_size):
            temp_instances = InstanceData()
            bboxes = torch.cat([ task_ret_list[b]['bboxes'] for task_ret_list in scene_ret[i][j] ]) # type: ignore
            bboxes = batch_single_input_metas[b]['box_type_3d'](bboxes,
                                                        self.base_model.pts_bbox_head.bbox_coder.code_size, # type: ignore
                                                        (0.5, 0.5, 0.5))
            scores = torch.cat([ task_ret_list[b]['scores'] for task_ret_list in scene_ret[i][j] ]) # type: ignore
            flag = 0
            for n, num_class in enumerate(self.base_model.pts_bbox_head.num_classes): # type: ignore
                scene_ret[i][j][n][b]['labels'] += flag # type: ignore
                flag += num_class
            labels = torch.cat([ task_ret_list[b]['labels'].int() for task_ret_list in scene_ret[i][j] ]) # type: ignore
            temp_instances.bboxes_3d = bboxes
            temp_instances.scores_3d = scores
            temp_instances.labels_3d = labels
            ret_list.append(temp_instances) # batch list of pred result instances
        # new_batch_data_samples = self.base_model.add_pred_to_datasample(batch_single_samples,
        #                                                                 ret_list,
        #                                                                 None)
        example_seq[i][j]['single_samples'] = self.base_model.add_pred_to_datasample(batch_single_samples,
                                                                    ret_list,
                                                                    None)