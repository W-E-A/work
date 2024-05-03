import torch
import os
import time
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmengine import mkdir_or_exist
import subprocess

def get_nvidia_smi_output():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running nvidia-smi:", e)
        return None

@HOOKS.register_module()
class ShowGPUMessage(Hook):

    def __init__(self, interval, log_level, log_dir, **kwargs):
        self.interval = interval
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        mkdir_or_exist(log_dir)
        log_file = os.path.join(log_dir, f'{timestamp}.log')
        log_cfg = dict(log_level=log_level, log_file=log_file, **kwargs)
        log_cfg.setdefault('name', self.__class__.__name__)
        log_cfg.setdefault('file_mode', 'a')
        self.logger = MMLogger.get_instance(**log_cfg)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """All subclasses should override this method, if they need any
        operations after each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        if self.every_n_train_iters(runner, self.interval) and torch.cuda.is_available():
            nvidia_smi_output = get_nvidia_smi_output()
            if nvidia_smi_output:
                self.logger.warn(f"GPU usage at {batch_idx} iter:")
                self.logger.warn(nvidia_smi_output)