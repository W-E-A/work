import argparse
import os
import os.path as osp
import json
import tqdm
import pdb

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

# DATA_ROOT = '/ai/volume/dataset/V2X-Seq-SPD'
# LABEL_EXT = '.json'

def parse_args():
    parser = argparse.ArgumentParser(description='Debug Program')
    parser.add_argument('config', help='config file path')
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


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    data_loader = Runner.build_dataloader(cfg.train_dataloader)
    item = next(iter(data_loader))
    pdb.set_trace()
    

if __name__ == '__main__':
    # label_root = osp.join(DATA_ROOT, 'cooperative/label')
    # label_list = os.listdir(label_root)
    # print(len(label_list))
    # cls_names = set()
    # for idx, label_name in tqdm.tqdm(enumerate(label_list)):
    #     with open(osp.join(label_root, label_name)) as f:
    #         label_dicts = json.load(f)
    #     for label_dict in label_dicts:
    #         cls_name = label_dict['type']
    #         cls_names.add(cls_name)
    # print(cls_names)
    main()