from typing import Any, Optional, Sequence
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS


@METRICS.register_module()
class V2XMetric(BaseMetric):
    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        self.default_prefix = 'V2X metric'
        super(V2XMetric, self).__init__(collect_device, prefix)

    
    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        return super().process(data_batch, data_samples)
    
    
    def compute_metrics(self, results: list) -> dict:
        return super().compute_metrics(results)