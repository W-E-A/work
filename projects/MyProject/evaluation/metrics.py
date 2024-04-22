# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log

from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes
from .utils import DeepAccident_det_eval


@METRICS.register_module()
class KittiMetricModified(BaseMetric):
    def __init__(self,
                 metric: Union[str, List[str]] = 'iou_mAP',
                 with_velocity: bool = True,
                 prefix: Optional[str] = None,
                 pklfile_prefix: Optional[str] = None,
                 format_only: bool = False,
                 submission_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'Kitti metric'
        super(KittiMetricModified, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.pklfile_prefix = pklfile_prefix
        self.format_only = format_only
        self.with_velocity = with_velocity
        if self.format_only:
            assert submission_prefix is not None, 'submission_prefix must be '
            'not None when format_only is True, otherwise the result files '
            'will be saved to a temp directory which will be cleaned up at '
            'the end.'

        self.submission_prefix = submission_prefix
        self.backend_args = backend_args

        allowed_metrics = ['iou_mAP', 'distance_mAP']
        self.metrics = metric if isinstance(metric, list) else [metric]
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(f"metric should be one of {allowed_metrics}, 'but got {metric}.'")

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            
            result = dict()

            pred_3d = data_sample['pred_instances_3d']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d

            gt_3d = data_sample['gt_instances_3d']
            for attr_name in gt_3d:
                gt_3d[attr_name] = gt_3d[attr_name].to('cpu')
            result['gt_instances_3d'] = gt_3d

            # result['scene_sample_idx'] = data_sample['scene_sample_idx']
            # result['scene_name'] = data_sample['scene_name']
            # result['agent_name'] = data_sample['agent_name']
            result['sample_idx'] = data_sample['sample_idx']
            result['box_type_3d'] = data_sample['box_type_3d']

            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        self.classes = self.dataset_meta['classes'] # type: ignore load from checkpoint

        result_dict, tmp_dir = self.format_results(
            results,
            pklfile_prefix=self.pklfile_prefix,
            submission_prefix=self.submission_prefix,
            classes=self.classes)

        metric_dict = {}

        if self.format_only:
            logger.info(
                f'results are saved in {osp.dirname(self.submission_prefix)}') # type: ignore
            return metric_dict

        gt_annos = result_dict['gt_instances_3d']
        dt_annos = result_dict['pred_instances_3d']

        for metric in self.metrics:
            ap_result_str, ap_dict = DeepAccident_det_eval(
                gt_annos,
                dt_annos,
                current_classes=self.classes,
                eval_types=metric)
            
            print_log('\n' + ap_result_str, logger=logger)

            metric_dict[metric] = ap_dict

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return metric_dict

    def format_results(
        self,
        results: List[dict],
        pklfile_prefix: Optional[str] = None,
        submission_prefix: Optional[str] = None,
        classes: Optional[List[str]] = None
    ) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_dict = dict()
        sample_idx_list = [result['sample_idx'] for result in results]
        for name in results[0]:
            if submission_prefix is not None:
                submission_prefix_ = osp.join(submission_prefix, name)
            else:
                submission_prefix_ = None
            if pklfile_prefix is not None:
                pklfile_prefix_ = osp.join(pklfile_prefix, name) + '.pkl'
            else:
                pklfile_prefix_ = None
            if 'pred_instances' in name and '3d' in name and name[
                    0] != '_' and results[0][name]:
                net_outputs = [result[name] for result in results]
                result_list_ = self.bbox2result_kitti(net_outputs,
                                                      results,
                                                      sample_idx_list,
                                                      classes, # type: ignore
                                                      'pd',
                                                      pklfile_prefix_,
                                                      submission_prefix_)
                result_dict[name] = result_list_
            if 'gt_instances' in name and '3d' in name and name[
                    0] != '_' and results[0][name]:
                net_outputs = [result[name] for result in results]
                result_list_ = self.bbox2result_kitti(net_outputs,
                                                      results,
                                                      sample_idx_list,
                                                      classes, # type: ignore
                                                      'gt',
                                                      pklfile_prefix_,
                                                      submission_prefix_)
                result_dict[name] = result_list_
        return result_dict, tmp_dir

    def bbox2result_kitti(
            self,
            net_outputs: List[dict],
            ori_results: List[dict],
            sample_idx_list: List[int],
            class_names: List[str],
            mode: str = 'pd',
            pklfile_prefix: Optional[str] = None,
            submission_prefix: Optional[str] = None) -> List[dict]:
        assert mode in ('gt', 'pd')
        if submission_prefix is not None:
            mmengine.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting 3D prediction to KITTI format')
        for idx, pred_dicts in enumerate(mmengine.track_iter_progress(net_outputs)):
            sample_idx = sample_idx_list[idx]
            if mode == 'gt':
                box_dict = self.convert_valid_bboxes_gt(pred_dicts, sample_idx, ori_results[idx])
            elif mode == 'pd':
                box_dict = self.convert_valid_bboxes_pd(pred_dicts, sample_idx, ori_results[idx])    
            if self.with_velocity:
                anno = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': [],
                    'velocity': []
                }
            else:
                anno = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': [],
                }
            if len(box_dict['box3d_lidar']) > 0:
                if mode == 'gt':
                    box_preds_lidar = box_dict['box3d_lidar']
                    label_preds = box_dict['label_preds']

                    for box, label in zip(box_preds_lidar,
                                          label_preds):
                        anno['name'].append(class_names[int(label)])
                        anno['truncated'].append(0.0)
                        anno['occluded'].append(0)
                        anno['score'].append(0.0)
                        anno['alpha'].append(0.0)
                        anno['dimensions'].append(box[3:6])
                        anno['location'].append(box[:3])
                        anno['rotation_y'].append(box[6])
                        if self.with_velocity:
                            anno['velocity'].append(box[7:9])

                    anno = {k: np.stack(v) for k, v in anno.items()}
                elif mode == 'pd':
                    scores = box_dict['scores']
                    box_preds_lidar = box_dict['box3d_lidar']
                    label_preds = box_dict['label_preds']
                    pred_box_type_3d = box_dict['pred_box_type_3d']

                    for box, score, label in zip(box_preds_lidar,
                                                 scores,
                                                 label_preds):
                        anno['name'].append(class_names[int(label)])
                        anno['truncated'].append(0.0)
                        anno['occluded'].append(0)
                        if pred_box_type_3d == LiDARInstance3DBoxes:
                            # anno['alpha'].append(
                            #     -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                            anno['alpha'].append(np.arctan2(-box[1], box[0]))
                        else:
                            raise TypeError("Unsupported bbox type.")
                        anno['dimensions'].append(box[3:6])
                        anno['location'].append(box[:3])
                        anno['rotation_y'].append(box[6])
                        anno['score'].append(score)
                        if self.with_velocity:
                            anno['velocity'].append(box[7:9])

                    anno = {k: np.stack(v) for k, v in anno.items()}
            else:
                if self.with_velocity:
                    anno = {
                        'name': np.array([]),
                        'truncated': np.array([]),
                        'occluded': np.array([]),
                        'alpha': np.array([]),
                        'dimensions': np.zeros([0, 3]),
                        'location': np.zeros([0, 3]),
                        'rotation_y': np.array([]),
                        'score': np.array([]),
                        'velocity': np.array([])
                    }
                else:
                    anno = {
                        'name': np.array([]),
                        'truncated': np.array([]),
                        'occluded': np.array([]),
                        'alpha': np.array([]),
                        'dimensions': np.zeros([0, 3]),
                        'location': np.zeros([0, 3]),
                        'rotation_y': np.array([]),
                        'score': np.array([])
                    }

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)

            if mode == 'pd':
                anno['sample_idx'] = np.array(
                    [sample_idx] * len(anno['score']), dtype=np.int64)

            det_annos.append(anno)

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            else:
                out = pklfile_prefix
            mmengine.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def convert_valid_bboxes_pd(self, box_dict: dict, sample_idx: int, metainfo: dict) -> dict:
        # FIXME: refactor this function
        box_preds = box_dict['bboxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        box_type_3d = metainfo['box_type_3d']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        if isinstance(box_preds, LiDARInstance3DBoxes):
            box_preds_lidar = box_preds
        else:
            raise TypeError("Unsupported bbox type.")

        return dict(
            pred_box_type_3d=box_type_3d,
            box3d_lidar=box_preds_lidar.tensor.numpy(),
            scores=scores.numpy(),
            label_preds=labels.numpy(),
            sample_idx=sample_idx)
    
    def convert_valid_bboxes_gt(self, box_dict: dict, sample_idx: int, metainfo: dict) -> dict:
        # FIXME: refactor this function
        box_preds = box_dict['bboxes_3d']
        labels = box_dict['labels_3d']
        box_type_3d = metainfo['box_type_3d']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        if isinstance(box_preds, LiDARInstance3DBoxes):
            box_preds_lidar = box_preds
        else:
            raise TypeError("Unsupported bbox type.")

        return dict(
            pred_box_type_3d=box_type_3d,
            box3d_lidar=box_preds_lidar.tensor.numpy(),
            label_preds=labels.numpy(),
            sample_idx=sample_idx)