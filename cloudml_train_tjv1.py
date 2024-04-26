import os
import re
import argparse
import copy


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--action",
                        default="crh-s",
                        help="{s:submit, ddd:delete, l:logs}")
    parser.add_argument(
        "--config",
        default=
        'projects/configs/bevformer/bevformer_tiny.py',
        help="train config file path")
    parser.add_argument("--job_name",
                        default='sparse4d-1',
                        help="name of your job")
    parser.add_argument(
        "--node",
        default='a3',
        help="configure training node type and num. [(a/v)-(num)], "
             "a strands for a100, v stands for v100. num is the machine num")

    parser.add_argument("--branch",
                        default="master",
                        help="train branch")
    parser.add_argument(
        "--jceph",
        default="/wyc/",
        help="you jceph, work dir will be your_jceph/job_name. training log and checkpoint will be saved here "
             "And Do not use zx-jceph1")
    parser.add_argument('--pro',
                        action='store_true',
                        help='use projects/tools/dist_train.sh')
    parser.add_argument('--joint', action='store_true', help='use joint_train')
    parser.add_argument('--test',
                        action='store_true',
                        help='use tools/dist_test.sh')
    parser.add_argument("--checkpoint", default='/mnt/ad-infra-dataset-pilot-fusion-team/checkpoints/cyl/train/cyl-det3d-0416/epoch_6.pth', help="chpt file path")
    parser.add_argument('--batch_size',
                        type=int,
                        default='1',
                        help='batch_size')
    parser.add_argument('--gpus_per_node',
                        type=int,
                        default='1',
                        help='')
    parser.add_argument('--tta',
                        type=int,
                        default='-1',
                        )
    parser.add_argument('--out',
                        type=str,
                        default='',
                        )
    parser.add_argument('--data_env',
                        default="nc4",
                        choices=["nc4", "c3"],
                        help="data environment")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.action == 'crh-s':  # 提交
        # common configs
        if ':' in args.jceph:
            jceph_root = args.jceph.split(':')[1]
        else:
            jceph_root = args.jceph.strip('/')
        tmp_dir = "/{}/tmp/{}".format(args.jceph, args.job_name)
        env = 'wyc-deep'

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

        # if args.test and args.deterministic_test:
        #     print('Cannot use test and deterministic_test at the same time.')
        #     exit()

        work_dir = "/{}/train/{}".format(jceph_root, args.job_name)
        test_checkpoint = ''
        if args.test:  # test
            program = "bash tools/dist_test.sh"
            test_checkpoint = args.checkpoint

            cfg_options = f"work_dir={work_dir} data.test.samples_per_gpu={args.batch_size}"


            other_args = f"--work-dir {work_dir}"
            test_command = f"{program} {pack_verion} 8 {args.config} {other_args}"
        else:  # train
            program = 'bash tools/dist_train.sh'
            # import pdb;pdb.set_trace()
            # cfg_options = f"work_dir={work_dir} data.samples_per_gpu={args.batch_size}"
            other_args = f"--work-dir {work_dir}"

            train_command = f"{program} {pack_verion} 8 {args.config} {other_args}"
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
                        -p {env} \
                        -dc " mkdir data \
                            && ln -s /mnt/infra_dataset_ssd/ad_infra_dataset_pilot_fusion/checkpoints/{jceph_root} /{jceph_root} \
                            && cd /mnt/auto-labeling/wyc/wyc-motion/deepaccident \
                            && git checkout dev-wyc-motion \
                            && git pull \
                            && source /root/miniconda3/bin/activate \
                            && nvidia-smi \
                            && conda activate py38t191 \
                            && ls . \
                            && python setup.py develop \
                            && export NCCL_IB_DISABLE=1 \
                            && {train_command} "\
                        -pn {args.node} \
                        -nas {nas_mounts}'

    else:
        assert NotImplementedError("Not support {}".format(args.action))

    print(command)
    os.system(command)


if __name__ == '__main__':
    main()
