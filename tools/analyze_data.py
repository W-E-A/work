import argparse
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import VISUALIZERS, DATASETS, TRANSFORMS
from mmengine.fileio import load, dump
from mmdet3d.utils import register_all_modules
from mmcv.transforms.base import BaseTransform
from pprint import pformat
import os
import time
import logging
import copy
import numpy as np
from projects.MyProject import SimpleLocalVisualizer
from to_gif import to_gif


def log(msg = "" ,level: int = logging.INFO):
    print_log(msg, "current", level)


def parse_args():
    parser = argparse.ArgumentParser(description='Data Analysis')
    parser.add_argument('config')
    parser.add_argument('--vis_save_path',
                        type=str,
                        default='data/analyze_output',
                        help='the dir to save vis data')
    parser.add_argument('--mode', type=str ,choices=['check_raw_info_format'], default='check_raw_info_format')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def check_raw_info_format(cfg, save_path, verbose):
    raw_info_save_path = os.path.join(save_path, 'raw_info.json')

    train_ann_file_path = cfg.train_annfile_path
    val_ann_file_path = cfg.val_annfile_path

    ann_file = load(val_ann_file_path)

    log(ann_file.keys())
    info = ann_file['data_list'][0]
    log(info.keys())

    dump(info, raw_info_save_path)
    
    if verbose:
        log()
        log(pformat(info))


def build_dataset_like_runner(dataset_cfg):
    # build dataset
    if isinstance(dataset_cfg, dict):
        dataset = DATASETS.build(dataset_cfg)
        if hasattr(dataset, 'full_init'):
            dataset.full_init()
    else:
        # fallback to raise error in dataloader
        # if `dataset_cfg` is not a valid type
        dataset = dataset_cfg
    return dataset


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    vis_save_path = os.path.join(
        os.path.abspath(args.vis_save_path),
        time.strftime(
            '%Y%m%d_%H%M%S',
            time.localtime(time.time())
        )
    )

    os.makedirs(vis_save_path, exist_ok=False)

    eval(args.mode)(cfg, vis_save_path, args.verbose)


if __name__ == '__main__':
    register_all_modules()
    main()
