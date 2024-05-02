import os
import os.path as osp
from glob import glob
import copy

import mmengine
import numpy as np
import multiprocessing as mp
import threading
import queue
from tqdm import tqdm
from time import sleep

deepaccident_categories = ('car', 'van', 'truck', 'cyclist', 'motorcycle', 'pedestrian')
deepaccident_folder_struct = ('split_txt_files', 'type1_subtype1_accident', 'type1_subtype1_normal', 'type1_subtype2_accident', 'type1_subtype2_normal')
scenario_types = ('type1_subtype1_accident','type1_subtype1_normal','type1_subtype2_accident','type1_subtype2_normal')
agent_names = ('ego_vehicle', 'other_vehicle', 'ego_vehicle_behind', 'other_vehicle_behind', 'infrastructure')
contents = {
    'BEV_instance_camera':'.npz',
    'Camera_Back':'.jpg',
    'Camera_BackLeft':'.jpg',
    'Camera_BackRight':'.jpg',
    'Camera_Front':'.jpg',
    'Camera_FrontLeft':'.jpg',
    'Camera_FrontRight':'.jpg',
    'calib':'.pkl',
    'label':'.txt',
    'lidar01':'.npz',
}

def _check_dataset_folder(data_path, struct):
    dirs = os.listdir(data_path)
    total = len(struct)
    for dir in dirs:
        if dir in struct:
            total -= 1
    assert (total == 0),f"Dataset is not well formatted and requires the following structure: {struct}"

def _read_list_file(root_path, prefix, ext = '.txt', split = ''):
    with open(osp.join(root_path, prefix+ext), 'r') as f:
        if split != '':
            items = [tuple(line.strip('\n').split(split)) for line in f]
        else:
            items = [line.strip('\n') for line in f]
    return items


def _process_target_items(target_items_subset, data_path, load_anno, sample_interval, pos, num_threads: int = 2):
    total_scene_name = []
    for scenario_type, scenario in tqdm(target_items_subset, desc=f"preparing - {pos}", position=pos, leave=False):
        total_scene_name.append('_'.join([scenario_type,scenario.split('_')[0], scenario.split('_')[-1]]))
    scene_data = {scene_name : {agent : [] for agent in agent_names} for scene_name in total_scene_name}
    for scenario_type, scenario in tqdm(target_items_subset, desc=f"processing - {pos}", position=pos, leave=False):
        assert scenario_type in scenario_types
        seq = [
            (int(osp.basename(frame).split('.')[0].split('_')[-1]), osp.basename(frame).split('.')[0])
            for frame in glob(osp.join(data_path,scenario_type,'ego_vehicle','label',scenario,'*'))]
        seq.sort(key= lambda x: x[0])
        last_seq_idx = seq[-1][0]
        filtered_seq = [frame[1] for frame in seq]
        
        meta = _read_list_file(osp.join(data_path,scenario_type,'meta'),scenario)
        meta_info = {}
        meta_info['sim_scenario'] = meta[0].split(' ') # type: ignore
        for line in meta[1:]:
            data = line.split(':') # type: ignore
            key = data[0]
            values = data[1].lstrip(' ').split(' ')
            meta_info[key] = values

        # parallel
        task_queue = queue.Queue()
        num_threads = num_threads  # 设置线程数量

        def process_frame(agent, frame, data_path, scenario_type, scenario, contents, load_anno, scene_data, meta_info, sample_interval, last_seq_idx):
            calib_path = osp.join(data_path,scenario_type,agent,'calib',scenario,frame+contents['calib'])
            calib_dict = mmengine.load(calib_path)

            cams = {}
            cam_name = [name for name in contents.keys() if name.startswith('Camera')]
            for name in cam_name:
                cam_info = {}
                cam_info['type'] = name
                cam_info['data_path'] = osp.join(osp.abspath(data_path),scenario_type,agent,name,scenario,frame+contents[name])
                cam_info['lidar_to_camera_matrix'] = calib_dict['lidar_to_'+name]
                camera_to_lidar_matrix = np.linalg.inv(calib_dict['lidar_to_'+name])
                cam_info['camera_to_lidar_matrix'] = camera_to_lidar_matrix
                cam_info['camera_to_ego_matrix'] = calib_dict['lidar_to_ego'] @ camera_to_lidar_matrix
                cam_info['ego_to_world_matrix'] = calib_dict['ego_to_world']
                cam_info['cam_intrinsic'] = calib_dict['intrinsic_'+name]
                cam_info['timestamp'] = int(frame.split('_')[-1])
                cams[name] = cam_info
            labels = _read_list_file(osp.join(data_path,scenario_type,agent,'label',scenario),frame,contents['label'])
            agent_speed_x = float(labels[0].split(' ')[0]) # type: ignore
            agent_speed_y = float(labels[0].split(' ')[1]) # type: ignore
            if load_anno:
                bbox_list = []
                valid_flag = []
                for label in labels[1:]:
                    if len(label.split(' ')) <= 1: # type: ignore
                        continue
                    cls_label = str(label.split(' ')[0]) # type: ignore
                    bbox = label.split(' ')[1:8] # type: ignore
                    bbox = list(map(float, bbox))
                    # ori box: xyz lwh mmstandyaw

                    vel = label.split(' ')[8:10] # type: ignore
                    vel = list(map(float, vel))
                    # reverse the y axis
                    vel = [vel[0], -vel[1]]
                    agent_id = int(label.split(' ')[-3]) # type: ignore
                    num_lidar_pts = int(label.split(' ')[-2]) # type: ignore
                    if label.split(' ')[-1] == 'True': # type: ignore
                        camera_visibility = 1
                    else:
                        camera_visibility = 0
                    bbox_list.append((cls_label,bbox,vel,agent_id,num_lidar_pts,camera_visibility))
                    if cls_label == 'pedestrian' or cls_label == 'motorcycle' or cls_label == 'cyclist':
                        if agent == 'infrastructure':
                            if num_lidar_pts <= 0:
                                valid_flag.append(False)
                            else:
                                valid_flag.append(True)
                        else:
                            if num_lidar_pts <= 1:
                                valid_flag.append(False)
                            else:
                                valid_flag.append(True)
                    else:
                        if agent == 'infrastructure':
                            if num_lidar_pts <= 0:
                                valid_flag.append(False)
                            else:
                                valid_flag.append(True)
                        else:
                            if num_lidar_pts <= 4:
                                valid_flag.append(False)
                            else:
                                valid_flag.append(True)

                label_array = np.array([bbox[0] for bbox in bbox_list]).reshape(-1)
                bbox_array = np.array([bbox[1] for bbox in bbox_list]).reshape(-1, 7)
                vel_array = np.array([bbox[2] for bbox in bbox_list]).reshape(-1, 2)
                vehicle_id_array = np.array([bbox[3] for bbox in bbox_list]).reshape(-1)
                num_lidar_pts_array = np.array([bbox[4] for bbox in bbox_list]).reshape(-1)
                camera_visibility_array = np.array([bbox[5] for bbox in bbox_list]).reshape(-1)
                valid_flag = np.array(valid_flag).reshape(-1)
            
            scenario_agent_frame_info = {} # 序列中的一帧
            scenario_agent_frame_info['scenario_meta'] = meta_info # 场景元数据
            scenario_agent_frame_info['scenario_type'] = scenario_type # 场景类型
            scenario_agent_frame_info['vehicle_name'] = agent # 协同对象名称
            scenario_agent_frame_info['scene_name'] = '_'.join([scenario_type,scenario.split('_')[0],scenario.split('_')[-1]]) # 场景名称
            scenario_agent_frame_info['lidar_prefix'] = '_'.join([scenario_type,agent,frame]) # 雷达帧标识符
            scenario_agent_frame_info['lidar_path'] = osp.join(osp.abspath(data_path),scenario_type,agent,'lidar01',scenario,frame+contents['lidar01']) # 雷达路径
            scenario_agent_frame_info['num_features'] = 4
            scenario_agent_frame_info['bev_path'] = osp.join(osp.abspath(data_path),scenario_type,agent,'BEV_instance_camera',scenario,frame+contents['BEV_instance_camera']) # BEV实例路径
            scenario_agent_frame_info['timestamp'] = int(frame.split('_')[-1]) # 帧时间戳，帧间隔为0.1s
            scenario_agent_frame_info['scenario_length'] = last_seq_idx # 最大时间戳
            scenario_agent_frame_info['cams'] = dict()# 多相机
            scenario_agent_frame_info['cams'].update(cams) # 多个相机，每个相机包括图像路径，标定参数
            scenario_agent_frame_info['lidar_to_ego_matrix'] = calib_dict['lidar_to_ego'] # 雷达外参
            scenario_agent_frame_info['ego_to_world_matrix'] = calib_dict['ego_to_world'] # 对象的世界坐标
            lidar_to_ego_matrix = np.array(calib_dict['lidar_to_ego'], dtype=np.float32)
            ego_to_world_matrix = np.array(calib_dict['ego_to_world'], dtype=np.float32)
            lidar_to_world_matrix = ego_to_world_matrix @ lidar_to_ego_matrix
            scenario_agent_frame_info['lidar_to_world_matrix'] = lidar_to_world_matrix
            scenario_agent_frame_info['sample_interval'] = sample_interval # 采样频率，决定帧间隔，原文为0.5s也就是间隔5采样
            scenario_agent_frame_info['vehicle_speed_x'] = agent_speed_x # 对象的x速度 # type: ignore
            scenario_agent_frame_info['vehicle_speed_y'] = agent_speed_y # 对象的y速度 # type: ignore
            if load_anno:
                scenario_agent_frame_info['gt_names'] = label_array # 世界类别gt # type: ignore
                scenario_agent_frame_info['gt_boxes'] = bbox_array # 世界boxes # type: ignore
                scenario_agent_frame_info['gt_velocity'] = vel_array# 世界速度gt # type: ignore
                scenario_agent_frame_info['vehicle_id'] = vehicle_id_array # 世界类别id # type: ignore
                scenario_agent_frame_info['num_lidar_pts'] = num_lidar_pts_array # 世界点云数目（目标上的点数） # type: ignore
                scenario_agent_frame_info['camera_visibility'] = camera_visibility_array # 世界目标是否可见 # type: ignore
                scenario_agent_frame_info['valid_flag'] = valid_flag # 雷达是否可见 # type: ignore
            scene_data[scenario_agent_frame_info['scene_name']][scenario_agent_frame_info['vehicle_name']].append(scenario_agent_frame_info)

        def worker():
            while True:
                agent, frame, data_path, scenario_type, scenario, contents, load_anno, scene_data, meta_info, sample_interval, last_seq_idx = task_queue.get()
                process_frame(agent, frame, data_path, scenario_type, scenario, contents, load_anno, scene_data, meta_info, sample_interval, last_seq_idx)
                task_queue.task_done()

        # 启动线程
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()

        # 将任务添加到队列
        for agent in agent_names:
            for frame in filtered_seq:
                task_queue.put((agent, frame, data_path, scenario_type, scenario, contents, load_anno, scene_data, meta_info, sample_interval, last_seq_idx))

        # 等待所有任务完成
        task_queue.join()
    
    return scene_data


def _get_detail_info(target_items, data_path, sample_interval, load_anno: bool = True, max_sweeps: int = 0, debug: bool = False):
    result = []
    if load_anno:
        # correct the velocity
        sample_dt = 0.1

    # parellel
    num_processes = max(os.cpu_count() // 8, 1)
    num_threads = os.cpu_count() // 2
    if debug:
        target_items = target_items[:10]
        num_processes = 5
    target_items_split = [target_items[i::num_processes] for i in range(num_processes)]

    with mp.Pool(processes=num_processes) as pool:
        scene_data_list = list(pool.starmap(_process_target_items, [(target_items_subset, data_path, load_anno, sample_interval, pos, num_threads) for pos, target_items_subset in enumerate(target_items_split)]))

    from collections import ChainMap
    scene_data = dict(ChainMap(*scene_data_list))

    for scene_name, agent_data_dict in tqdm(scene_data.items(), desc=f"generate velocity data...", leave=False): # 用多线程跑
        for agent in agent_names:
            if len(agent_data_dict[agent]) <= 0:
                print(f"{scene_name} -> {agent} seq empty, skip it!!!")
                continue
            if load_anno:
                agent_data_dict[agent] = sorted(agent_data_dict[agent], key=lambda x: x['timestamp'])
                data_len = len(agent_data_dict[agent])
                ego_motion = np.array([data['ego_to_world_matrix'] for data in agent_data_dict[agent]], dtype=np.float32) # N 4 4
                ego_xy = ego_motion[:, :2, 3] # N 2
                ego_xy_delta = ego_xy[1:] - ego_xy[:-1] # N-1 2
                ego_vel = ego_xy_delta / sample_dt # N 2
                ego_vel = np.concatenate([ego_vel[0][None, :], ego_vel], axis=0) # N 2

                id_set = set([id for info in agent_data_dict[agent] for id in info['vehicle_id']])
                track_map = {id : [] for id in id_set}
                for idx, info in enumerate(agent_data_dict[agent]):
                    info['vehicle_speed_x'] = ego_vel[idx][0]
                    info['vehicle_speed_y'] = ego_vel[idx][1]
                    for _, v in track_map.items():
                        v.append(None)
                    for jdx, id in enumerate(info['vehicle_id']):
                        box_center = copy.deepcopy(info['gt_boxes'][jdx][:4])
                        box_center[-1] = 1
                        track_map[id][-1] = box_center

                track_vel = {}
                for id, centers in track_map.items():

                    trj_centers_world = []
                    ego_rots = [] # trj 2 2
                    for i, center in enumerate(centers):
                        if center is not None:
                            ego_rots.append(ego_motion[i][:2, :2])
                            center_world = ego_motion[i] @ center # 4 4 @ 4, 
                            trj_centers_world.append(center_world[:2]) # 2
                        else:
                            ego_rots.append(None)
                            trj_centers_world.append(None)

                    total_len = len(trj_centers_world)
                    trj_centers_ground_vels = []
                    start = False
                    start_fill = False
                    pending = 1
                    for i in range(1, total_len):
                        if not start and trj_centers_world[i-1] is not None:
                            start = True
                        if start:
                            pending_idx = i - pending
                            if trj_centers_world[i] is None:
                                if i == total_len - 1:
                                    if not start_fill:
                                        trj_centers_ground_vels.append(np.array([0.0, 0.0], dtype=np.float32))
                                        start_fill = True
                                    for _ in range(pending):
                                        trj_centers_ground_vels.append(None)
                                    break # stop the loop
                                pending += 1
                                continue
                            if pending > 2:
                                if not start_fill:
                                    trj_centers_ground_vels.append(np.array([0.0, 0.0], dtype=np.float32))
                                    start_fill = True
                                for _ in range(pending-1):
                                    trj_centers_ground_vels.append(None)
                                start = False
                                start_fill = False
                                if i == total_len - 1:
                                    trj_centers_ground_vels.append(np.array([0.0, 0.0], dtype=np.float32))
                                    break  # stop the loop
                            else:
                                trj_centers_world_delta = trj_centers_world[i] - trj_centers_world[pending_idx]
                                trj_centers_world_vel = trj_centers_world_delta / sample_dt / pending
                                if not start_fill:
                                    trj_centers_ground_vels.append(trj_centers_world_vel @ ego_rots[pending_idx])
                                    start_fill = True
                                for _ in range(pending-1):
                                    trj_centers_ground_vels.append(None)
                                trj_centers_ground_vels.append(trj_centers_world_vel @ ego_rots[i])
                            pending = 1
                        else:
                            trj_centers_ground_vels.append(None)
                            if i == total_len - 1: # final stop
                                trj_centers_ground_vels.append(np.array([0.0, 0.0], dtype=np.float32))
                        
                    track_vel[id] = trj_centers_ground_vels
                
                for idx, info in enumerate(agent_data_dict[agent]):
                    try:
                        info['gt_velocity'] = np.array([track_vel[id][idx] for id in info['vehicle_id']], dtype=np.float32) # N 2
                    except:
                        import pdb
                        pdb.set_trace()

            timestamps = [x['timestamp'] for x in agent_data_dict[agent]]
            last_timestamp = timestamps[-1]
            filtered_idx = []
            for idx, timestamp in enumerate(timestamps):
                if (last_timestamp - timestamp) % sample_interval == 0:
                    filtered_idx.append(idx)
            assert len(filtered_idx) > 0,f"No sample after {sample_interval} filtered"
            if max_sweeps > 0:
                assert max_sweeps < sample_interval
                temp = []
                for idx in filtered_idx:
                    data = agent_data_dict[agent][idx]
                    key_to_world_matrix = agent_data_dict[agent][idx]['lidar_to_world_matrix']
                    sweeps = []
                    while (len(sweeps) < max_sweeps and idx > len(sweeps)):
                        sweep_to_world_matrix = agent_data_dict[agent][idx - len(sweeps) - 1]['lidar_to_world_matrix']
                        sweep = {
                            'lidar_path' : agent_data_dict[agent][idx - len(sweeps) - 1]['lidar_path'],
                            'timestamp' : agent_data_dict[agent][idx - len(sweeps) - 1]['timestamp'],
                            'lidar_to_ego_matrix' : agent_data_dict[agent][idx - len(sweeps) - 1]['lidar_to_ego_matrix'],
                            'sweep_to_key_matrix' : np.linalg.inv(key_to_world_matrix) @ sweep_to_world_matrix,
                        }
                        sweeps.append(sweep)
                    data['sweeps'] = sweeps
                    temp.append(data)
                agent_data_dict[agent] = temp
            else:
                agent_data_dict[agent] = [agent_data_dict[agent][idx] for idx in filtered_idx]
            result.extend(agent_data_dict[agent])
    return result


def create_deepaccident_info_file(data_path, pkl_prefix='deepaccident', save_path=None, sample_interval = 5, max_sweeps = 0, debug = False):
    assert sample_interval >= 1
    _check_dataset_folder(data_path, deepaccident_folder_struct)
    train_split_items = _read_list_file(osp.join(data_path,'split_txt_files'),'train',split=' ')
    val_split_items = _read_list_file(osp.join(data_path,'split_txt_files'),'val',split=' ')
    # assume test equals val
    test_split_items = _read_list_file(osp.join(data_path,'split_txt_files'),'val',split=' ')

    if save_path is None:
        save_path = data_path

    output_dict = {}
    output_dict['metainfo'] = {
        'classes': deepaccident_categories,
        'agents': agent_names
    }

    mmengine.print_log('Generate info. this may take several minutes.','current')
    deepaccident_infos_train = _get_detail_info(train_split_items,data_path,sample_interval,max_sweeps=max_sweeps, debug=debug)
    filename = osp.join(save_path,f'{pkl_prefix}_infos_train.pkl')
    output_dict['data_list'] = deepaccident_infos_train
    mmengine.dump(output_dict, filename)
    mmengine.print_log(f'DeepAccident info train file is saved to {filename}','current')
    deepaccident_infos_val = _get_detail_info(val_split_items,data_path,sample_interval,max_sweeps=max_sweeps, debug=debug)
    filename = osp.join(save_path,f'{pkl_prefix}_infos_val.pkl')
    output_dict['data_list'] = deepaccident_infos_val
    mmengine.dump(output_dict, filename)
    mmengine.print_log(f'DeepAccident info val file is saved to {filename}','current')
    # deepaccident_infos_test = _get_detail_info(test_split_items,data_path,sample_interval,max_sweeps=max_sweeps, debug=debug)
    # filename = osp.join(save_path,f'{pkl_prefix}_infos_test.pkl')
    # output_dict['data_list'] = deepaccident_infos_test
    # mmengine.dump(output_dict, filename)
    # mmengine.print_log(f'DeepAccident info test file is saved to {filename}, assume test equals val','current')

    
    