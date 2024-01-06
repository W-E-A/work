from typing import Any, Optional, Sequence
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS
from .models.utils import caluclate_tp_fp, eval_final_results


@METRICS.register_module()
class V2XMetric(BaseMetric):
    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        self.default_prefix = 'V2X metric'
        super(V2XMetric, self).__init__(collect_device, prefix)

        self.result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0},
                            0.5: {'tp': [], 'fp': [], 'gt': 0},
                            0.7: {'tp': [], 'fp': [], 'gt': 0}}
    
    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        pred_box = data_samples[0]
        pred_score = data_samples[1]
        gt_box = data_samples[2]
        caluclate_tp_fp(pred_box, pred_score, gt_box, self.result_stat, 0.3)
        caluclate_tp_fp(pred_box, pred_score, gt_box, self.result_stat, 0.5)
        caluclate_tp_fp(pred_box, pred_score, gt_box, self.result_stat, 0.7)
    
    
    def compute_metrics(self, results: list) -> dict:
        
        ap_30, ap_50, ap_70 = eval_final_results(self.result_stat)
        
        return {
            'AP@0.3' : ap_30,
            'AP@0.5' : ap_50,
            'AP@0.7' : ap_70
        }