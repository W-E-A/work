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


def log(msg = "" ,level: int = logging.INFO):
    print_log(msg, "current", level)


def parse_args():
    parser = argparse.ArgumentParser(description='Data Analysis')
    parser.add_argument('config')
    parser.add_argument('--vis_save_path',
                        type=str,
                        default='data/gt_vis_data',
                        help='the dir to save vis data')
    parser.add_argument('--mode', type=str ,choices=['check_raw_info_format', 'analyze_data'], default='check_raw_info_format')
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


@TRANSFORMS.register_module()
class CorrelationFilter(BaseTransform):

    def __init__(self,
        # pc_range,
        # voxel_size,
        ego_name: str = 'ego_vehicle',
        with_velocity: bool = True,
        only_vehicle: bool = False,
        vehicle_id_list: list = [0, 1, 2],
        ego_id: int = -100,
        min_distance_thres: float = 0.5,
        max_distance_thres: float = 15,
        alpha_coeff: float = 1,
        visualizer_cfg: dict = None,
        just_save_root: str = None,
        verbose: bool = False
    ) -> None:
        # self.pc_range = pc_range
        # self.voxel_size = voxel_size
        self.ego_name = ego_name
        self.with_velocity = with_velocity
        self.only_vehicle = only_vehicle
        self.vehicle_id_list = vehicle_id_list
        self.ego_id = ego_id
        self.min_distance_thres = min_distance_thres
        self.max_distance_thres = max_distance_thres
        self.alpha_coeff = alpha_coeff
        self.visualizer = None
        if visualizer_cfg is not None:
            self.visualizer: SimpleLocalVisualizer = VISUALIZERS.build(visualizer_cfg)
            assert just_save_root != None
        self.just_save_root = just_save_root
        self.verbose = verbose

        # self.voxel_size = np.array(self.voxel_size).astype(np.float32)

        # self.grid_size = np.array([
        #     np.ceil((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1]), # H
        #     np.ceil((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0]), # W
        #     np.ceil((self.pc_range[5] - self.pc_range[2]) / self.voxel_size[2]), # D
        # ]).astype(np.int32) # 1024 1024 1

        # self.offset_xy = np.array([
        #     self.pc_range[0] + self.voxel_size[0] * 0.5,
        #     self.pc_range[1] + self.voxel_size[1] * 0.5
        # ]).astype(np.float32)

        # if self.visualize:
        #     self.scatter_trans = np.array(
        #         [[-1, 0],
        #         [0, -1]],
        #         dtype=np.float32
        #     )
        #     self.save_count = 0
        #     self.fig, self.ax = plt.subplots(1, 1)

        self.cmaps = [
            'Oranges', 'Greens', 'Purples', 'Oranges', 'PuRd', 'BuPu',
        ]
        # self.cmaps = [
        #     'Greys', 'Purples', 'Greens', 'Oranges', 'Reds',
        #     'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        #     'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
        # ]

    def linear_correlation_score(self,
            ego_center: np.ndarray,
            target_center: np.ndarray,
            min_distance_thres: float,
            max_distance_thres: float,
            alpha_coeff: float
        ):
        assert ego_center.shape == target_center.shape
        assert len(ego_center.shape) == 2
        if len(ego_center) <= 0:
            return 0
        
        encounter_distance = np.linalg.norm(ego_center - target_center, axis=1)
        encounter_distance = np.clip(encounter_distance, min_distance_thres, max_distance_thres)
        base_score = np.interp(encounter_distance, [min_distance_thres, max_distance_thres], [1, 1e-4])
        time_decay_coeff = np.exp(-alpha_coeff * np.linspace(0, 1, len(base_score)))
        time_decay_coeff = time_decay_coeff / np.sum(time_decay_coeff)
        decay_score = time_decay_coeff * base_score
        final_score = np.sum(decay_score)

        return final_score

    def potential_score(self,
        ego_center: np.ndarray, # 2,
        ego_vel: np.ndarray, # 2,
        target_center: np.ndarray, # 2,
        target_vel: np.ndarray, # 2,
    ):
        assert ego_center.shape == target_center.shape == ego_vel.shape == target_vel.shape
        assert len(ego_center) == 2 and len(ego_center.shape) == 1
            
        r_vector = ego_center - target_center
        


        return 1.0


    def transform(self, input_dict):
        present_idx = input_dict['present_idx'] # 当前序列第几位开始是关键帧, 比如0
        seq_length = input_dict['seq_length'] # 当前序列长度，比如6
        scene_name = input_dict['scene_name'] # 序列所属场景
        seq_timestamp_0 = input_dict['seq_timestamps'][0] # 序列起始的时间戳，比如5
        co_agents = copy.deepcopy(input_dict['co_agents']) # 参与协同的代理名称
        example_seq = input_dict['example_seq'] # 经过单车pipline得到的输入和标签数据
        future_seq_timestamps = range(seq_length)[present_idx:] # 取出未来帧的位置， 比如3, 4, 5
        future_length = len(future_seq_timestamps) # 表明未来帧的长度，比如3
        future_pose_matrix = input_dict['pose_matrix'][present_idx:, ...] # x c c 4 4 # 未来帧的每帧变换关系
        future_motion_matrix = input_dict['future_motion_matrix'] # c x 4 4 未来帧相对于起始帧的位姿关系
        # future_loc_matrix = input_dict['loc_matrix'][present_idx:, ...] # x c 4 4 # 未来帧ego的实际位置

        co_agents = {v : k for k, v in enumerate(co_agents)} # str : int

        assert self.ego_name in co_agents and self.ego_name != 'infrastructure'
        track_map_list = []
        for agent, j in co_agents.items():
            if agent == self.ego_name:
                # 如果处理的是ego则先得到ego的规划轨迹和关键帧相对于其他协同对象的变换矩阵
                ego_motion_matrix = future_motion_matrix[j] # x 4 4 rela to present
                present_trans_matrix = future_pose_matrix[0, :, j] # c 4 4
                present_trans_matrix = np.delete(present_trans_matrix, j, axis=0) # ego to [ego, other, ...] at present c-1 4 4
                if self.with_velocity:
                    # ego2globals = []
                    ego_global_vels = []
                    for i, timestamp in enumerate(future_seq_timestamps):
                        ego_global_vels.append([
                            example_seq[timestamp][j]['data_samples'].metainfo['vehicle_speed_x'],
                            example_seq[timestamp][j]['data_samples'].metainfo['vehicle_speed_y']
                        ])
                        # ego2globals.append(np.linalg.inv(np.array(example_seq[timestamp][j]['data_samples'].metainfo['ego2global'])))
                    ego_global_vels = np.array(ego_global_vels, dtype=np.float32) # N, 2
                    # ego2globals_rots = np.array(ego2globals, dtype=np.float32) # N, 4, 4
                    # motion_rots = ego_motion_matrix[:, :3, :3]
                    global2ego_prescent_rots = np.linalg.inv(np.array(example_seq[present_idx][j]['data_samples'].metainfo['ego2global'], dtype=np.float32))[:2, :2]
                    ego_present_vels = ego_global_vels @ global2ego_prescent_rots.T # N, 2 ego的序列速度同一相对于present的结果

            else:
                # 如果处理的是协同代理，则对于每个代理在时序上跟踪所有目标，相对于关键帧的轨迹，每个代理得到一个track_map
                track_map = {}
                for i, timestamp in enumerate(future_seq_timestamps):
                    gt_instances_3d = example_seq[timestamp][j]['data_samples'].gt_instances_3d.clone()

                    if self.only_vehicle:
                        vehicle_mask = np.isin(gt_instances_3d.labels_3d, self.vehicle_id_list)
                        gt_instances_3d = gt_instances_3d[vehicle_mask]
                    
                    if agent != 'infrastructure':
                        ego_mask = gt_instances_3d.track_id != self.ego_id
                        gt_instances_3d = gt_instances_3d[ego_mask]

                    gt_instances_3d = gt_instances_3d[gt_instances_3d.bbox_3d_isvalid] # only visible target

                    lidar2agent = np.array(example_seq[timestamp][j]['data_samples'].metainfo['lidar2ego'], dtype=np.float32) # type: ignore
                    agent2present = future_motion_matrix[j][i]
                    lidar2present = lidar2agent @ agent2present
                    gt_instances_3d.bboxes_3d.rotate(lidar2present[:3, :3].T, None)
                    gt_instances_3d.bboxes_3d.translate(lidar2present[:3, 3]) # valid boxes to present
                    
                    for idx, id in enumerate(gt_instances_3d.track_id):
                        bboxes_xy = gt_instances_3d.bboxes_3d.tensor[idx].numpy()[:2] # agent坐标系下其他目标的present的中心
                        if self.with_velocity:
                            bboxes_vel = gt_instances_3d.bboxes_3d.tensor[idx].numpy()[-2:] # agent坐标系下其他目标的present的速度
                        labels = gt_instances_3d.labels_3d[idx]
                        if id not in track_map:
                            track_map[id] = {
                                # 'color': self.cmaps[np.random.randint(0, len(self.cmaps))],
                                'color': self.cmaps[labels],
                                'center': [],
                                'vel': [],
                                'start': i
                            }
                        track_map[id]['center'].append(bboxes_xy)
                        if self.with_velocity:
                            track_map[id]['vel'].append(bboxes_vel)
                track_map_list.append(track_map)

        ego_dix = co_agents.pop(self.ego_name) # ignore the ego

        for i, (agent, j) in enumerate(co_agents.items()): # agent
            # 对于每个协同代理的视角下，将ego的规划轨迹转到协同代理的当前坐标系下
            track_map = track_map_list[i]
            present_instances_3d = example_seq[present_idx][j]['data_samples'].gt_instances_3d
            trans = present_trans_matrix[i] # type: ignore
            rela_matrix = np.stack([trans @ ego_motion_matrix[k] for k in range(future_length)], axis=0) # type: ignore
            rela_centers = rela_matrix[:, :2, 3] # x 2
            if self.with_velocity:
                rots = trans[:2, :2]
                rela_vels = ego_present_vels @ rots.T # N, 2

            if self.visualizer:
                # rela_centers_vis = rela_centers @ self.scatter_trans.T # type: ignore
                self.visualizer.set_points_from_npz(example_seq[present_idx][j]['data_samples'].metainfo["lidar_path"])
            
            # 对于每个协同代理的检测结果（这里是真值），计算其和ego轨迹的关系
            result_dict = {}
            for id, v in track_map.items():
                # 对于每一条轨迹
                color = v['color'] # 类型颜色
                start = v['start'] # 开始跟踪的时间戳
                centers = np.stack(v['center'], axis=0) # N, 2 # 轨迹中心点
                cmp_centers = rela_centers[start:start + len(centers)] # N, 2 这里保证ego是从头开始的，因此在有效预测目标轨迹的时间段对比
                cmp_vels = rela_vels[start:start + len(centers)] # N, 2
                correlation_score = 0
                if self.with_velocity:
                    vels = np.stack(v['vel'], axis=0) # N, 2
                    correlation_score = self.linear_correlation_score(
                        cmp_centers, centers, self.min_distance_thres, self.max_distance_thres, self.alpha_coeff
                    )
                    potential_score = self.potential_score(
                        cmp_centers[0], cmp_vels[0], centers[0], vels[0] # N, 2 -> present frame info for potential score
                    )
                    correlation_score = correlation_score * potential_score
                else:
                    correlation_score = self.linear_correlation_score(
                        cmp_centers, centers, self.min_distance_thres, self.max_distance_thres, self.alpha_coeff
                    )

                vel_vector = np.stack([centers[0], centers[0] + 0.5 * vels[0]]) # 2, 2
                cmp_vel_vector = np.stack([cmp_centers[0], cmp_centers[0] + 0.5 * cmp_vels[0]]) # 2, 2

                result_dict[id] = correlation_score

                if self.visualizer:
                    self.visualizer.draw_points(vel_vector, colors='r', sizes = 100)
                    self.visualizer.draw_points(cmp_vel_vector, colors='g', sizes = 100)
                    # centers_vis = centers @ self.scatter_trans.T # type: ignore
                    self.visualizer.draw_points(centers, colors=np.linspace(0.0, 1.0, len(centers))[::-1], sizes = 40, cmap='Greens')
            
            correlation_scores = np.array([result_dict[id] if id in result_dict else 0 for id in present_instances_3d.track_id], dtype=np.float32)
            present_instances_3d['correlation'] = correlation_scores
            example_seq[present_idx][j]['data_samples'].gt_instances_3d = present_instances_3d
            if self.visualizer:
                self.visualizer.draw_points(rela_centers, colors=np.linspace(0.0, 1.0, future_length)[::-1], sizes = 80, cmap='Blues')
                self.visualizer.draw_bev_bboxes(present_instances_3d.bboxes_3d, c='#FF8000')
                correlation_scores_str = [f"{s:.2f}" for s in list(correlation_scores)]
                self.visualizer.draw_texts(
                    correlation_scores_str,
                    present_instances_3d.bboxes_3d.gravity_center[..., :2],
                    font_sizes = 15,
                    colors = 'orange',
                    vertical_alignments = 'center',
                    horizontal_alignments = 'center')

                if self.verbose:
                    log(result_dict)
                    log(correlation_scores)
                    log(agent)
                self.visualizer.just_save(os.path.join(self.just_save_root, f"in_{agent}_{time.time_ns()}.png"))
                self.visualizer.clean()

        example_seq[present_idx][ego_dix]['data_samples'].gt_instances_3d['correlation'] = \
        np.ones_like(example_seq[present_idx][ego_dix]['data_samples'].gt_instances_3d.track_id, dtype=np.float32) # type: ignore
        input_dict['example_seq'] = example_seq

        return input_dict


def analyze_data(cfg, save_path, verbose):
    train_dataset_cfg = cfg.train_dataloader.dataset
    test_dataset_cfg = cfg.test_dataloader.dataset

    train_scene_pipline = cfg.train_scene_pipline
    test_scene_pipline = cfg.test_scene_pipline

    train_ego_name = cfg.train_ego_name
    test_ego_name = cfg.test_ego_name

    dataset_cfg = test_dataset_cfg
    scene_pipline = test_scene_pipline
    ego_name = test_ego_name

    dataset_cfg.key_interval = 1
    dataset_cfg.seq_length = 6
    dataset_cfg.present_idx = 0
    dataset_cfg.co_agents = ('ego_vehicle', 'infrastructure')
    dataset_cfg.scene_shuffle = False
    dataset_cfg.with_velocity = True

    scene_pipline.pop(1)
    correlation_filter_cfg = dict(
        type = 'CorrelationFilter',
        ego_name = ego_name,
        with_velocity = dataset_cfg.with_velocity,
        only_vehicle = False,
        vehicle_id_list = [0, 1, 2],
        ego_id = -100,
        min_distance_thres = 0.5,
        max_distance_thres = 70,
        alpha_coeff = 1,
        visualizer_cfg = dict(
            type='SimpleLocalVisualizer',
            pc_range=cfg.lidar_range,
            voxel_size=cfg.voxel_size,
            name='visualizer',
        ),
        just_save_root = save_path,
        verbose = verbose
    )
    scene_pipline.insert(1, correlation_filter_cfg)
    dataset_cfg.scene_pipline = scene_pipline

    dataset = build_dataset_like_runner(dataset_cfg)
    dataset_len = len(dataset)

    for i in range(dataset_len):
        seq_data = dataset[i]

        if i > 40:
            break
    
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
