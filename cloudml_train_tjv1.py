import os
import argparse
import copy

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--action", default="crh-s", help="{s:submit, ddd:delete, l:logs}")
    parser.add_argument("--env", default='cyl-deep', help="remote project env name")
    parser.add_argument("--config", default='projects/MyProject/configs/exp_new_2_cloud.py', help="train config file path")
    parser.add_argument("--job-name", default='work_dirs', help="name of your job")
    parser.add_argument("--node", default='LF_R4A', help="configure training node type and num. [(a/v)-(num)], a strands for a100, v stands for v100. num is the machine num")
    parser.add_argument("--branch", default="dev-wea-cloud", help="train branch")
    parser.add_argument("--jceph", default="/cyl_deep/", help="you jceph, work dir will be your_jceph/job_name. training log and checkpoint will be saved here And Do not use zx-jceph1")
    parser.add_argument('--test', action='store_true', help='use tools/dist_test.sh')
    parser.add_argument("--checkpoint", default='', help="chpt file path")
    parser.add_argument("--gpus-per-node", default=8, help="gpus per node")
    parser.add_argument('--batch-size', type=int, default=1, help='batch size per gpu')
    parser.add_argument('--num-workers',type=int, default=32, help='num workers per gpu')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.action == 'crh-s':  # 提交
        if ':' in args.jceph:
            args.jceph = args.jceph.split(':')[1]
        else:
            args.jceph = args.jceph.strip('/')
        def print_node_num(pack_name):
            print('package name:', pack_name)
            if pack_name.startswith('FS_') or pack_name.startswith('LF_') or pack_name.startswith('PD_'):
                pack_name = copy.copy(pack_name)[3:]
            pack_verion = ''
            for i in range(1, len(pack_name)):
                if pack_name[i:i + 1].isdigit():
                    pack_verion += pack_name[i:i + 1]
                else:
                    break
            pack_verion = int(pack_verion)
            print('node num:', pack_verion)
            return pack_verion
        pack_verion = print_node_num(args.node)
        work_dir = "/{}/train/{}".format(args.jceph, args.job_name)
        if args.test:  # test
            pass
            # FIXME BUG Here
            # program = 'bash tools/dist_test.sh'
            # test_checkpoint = args.checkpoint
            # cfg_options = f"work_dir={work_dir} data.test.samples_per_gpu={args.batch_size}"
            # other_args = f"--work-dir {work_dir}"
            # test_command = f"{program} {pack_verion} {args.gpus_per_node} {args.config} {other_args}"
        else:  # train
            program = 'bash tools/dist_train.sh'
            other_args = f"--work-dir {work_dir} --batch-size {args.batch_size} --num-workers {args.num_workers}"
            train_command = f"{program} {pack_verion} {args.gpus_per_node} {args.config} {other_args}"
        nas_secret_to_mounts = {
            'auto-labeling-tj-org': '/mnt/auto-labeling',
            # 'fusion-team-frame-data-cache': '/mnt/frame-data-cache',
            # 'fusion-team-output-data-storage': '/mnt/output-data-storage',
            # 'fusion-team-algo-pilot-lidar': '/mnt/ad_algo/algo-pilot-lidar',
            # 'fusion-team-ad-infra-dataset-pilot-static': '/mnt/infra_dataset_ssd/ad_infra_dataset_pilot_static',
            'fusion-team-ad-infra-dataset-pilot-fusion': '/mnt/infra_dataset_ssd/ad_infra_dataset_pilot_fusion'
        }
        nas_mounts = ",".join([f'{k}:{v}' for k, v in nas_secret_to_mounts.items()])
        command = f'crh -k tjv1 jobs submit \
                        -n {args.job_name} \
                        -p {args.env} \
                        -dc " ln -s /mnt/infra_dataset_ssd/ad_infra_dataset_pilot_fusion/checkpoints/{args.jceph} /{args.jceph} \
                            && git checkout {args.branch} \
                            && git pull \
                            && source /root/miniconda3/bin/activate \
                            && nvidia-smi \
                            && conda activate py38t191 \
                            && pip install -i https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple mmcv==2.0.0rc4 --force-reinstall \
                            && ls . \
                            && export TORCH_CUDA_ARCH_LIST="7.5" \
                            && export NCCL_IB_DISABLE=1 \
                            && export PYTHONPATH=./ \
                            && pip install -i https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple numpy==1.22.0 \
                            && pip install -e . \
                            && {train_command} "\
                        -pn {args.node} \
                        -nas {nas_mounts}'
    else:
        assert NotImplementedError("Not support {}".format(args.action))
    print(command)
    os.system(command)
if __name__ == '__main__':
    main()
