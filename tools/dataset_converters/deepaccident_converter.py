import os
import os.path as osp
from glob import glob

import mmengine
import numpy as np

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


def _read_list_file(root_path, prefix, ext = '.txt', split = ''):
    with open(osp.join(root_path, prefix+ext), 'r') as f:
        if split != '':
            items = [tuple(line.strip('\n').split(split)) for line in f]
        else:
            items = [line.strip('\n') for line in f]
    return items


def _get_detail_info(target_items, data_path, sample_interval):
    result = []
    for scenario_type, scenario in mmengine.track_iter_progress(target_items):
        assert scenario_type in scenario_types
        seq = [
            (int(osp.basename(frame).split('.')[0].split('_')[-1]), osp.basename(frame).split('.')[0])
              for frame in glob(osp.join(data_path,scenario_type,'ego_vehicle','label',scenario,'*'))]
        seq.sort(key= lambda x: x[0])
        last_seq_idx = seq[-1][0]
        filtered_seq = []
        for frame in seq:
            if (last_seq_idx - frame[0]) % sample_interval == 0:
                filtered_seq.append(frame[1])
        assert len(filtered_seq) > 0,f"No sample after {sample_interval} filtered"
        
        meta = _read_list_file(osp.join(data_path,scenario_type,'meta'),scenario)
        meta_info = {}
        meta_info['sim_scenario'] = meta[0].split(' ') # type: ignore
        for line in meta[1:]:
            data = line.split(':') # type: ignore
            key = data[0]
            values = data[1].lstrip(' ').split(' ')
            meta_info[key] = values
        
        for agent in agent_names:
            for frame in filtered_seq:
                
                calib_path = osp.join(data_path,scenario_type,agent,'calib',scenario,frame+contents['calib'])
                calib_dict = mmengine.load(calib_path)

                cams = {}
                cam_name = [ name for name in contents.keys() if name.startswith('Camera')]
                for name in cam_name:
                    cam_info = {}
                    cam_info['image_path'] = osp.join(osp.abspath(data_path),scenario_type,agent,name,scenario,frame+contents[name])
                    cam_info['lidar_to_camera_matrix'] = calib_dict['lidar_to_'+name]
                    cam_info['camera_intrinsic_matrix'] = calib_dict['intrinsic_'+name]
                    cam_info['timestamp'] = int(frame.split('_')[-1])
                    cams[name] = cam_info

                labels = _read_list_file(osp.join(data_path,scenario_type,agent,'label',scenario),frame,contents['label'])
                agent_speed_x = float(labels[0].split(' ')[0]) # type: ignore
                agent_speed_y = float(labels[0].split(' ')[1]) # type: ignore
                bbox_list = []
                for label in labels[1:]:
                    if len(label.split(' ')) <= 1: # type: ignore
                        continue
                    cls_label = str(label.split(' ')[0]) # type: ignore
                    bbox = label.split(' ')[1:8] # type: ignore
                    bbox = list(map(float, bbox))
                    ##
                    # xyz hwl yaw
                    bbox[6] = -bbox[6]
                    ##
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

                label_array = np.array([bbox[0] for bbox in bbox_list]).reshape(-1)
                bbox_array = np.array([bbox[1] for bbox in bbox_list]).reshape(-1, 7)
                vel_array = np.array([bbox[2] for bbox in bbox_list]).reshape(-1, 2)
                vehicle_id_array = np.array([bbox[3] for bbox in bbox_list]).reshape(-1)
                num_lidar_pts_array = np.array([bbox[4] for bbox in bbox_list]).reshape(-1)
                camera_visibility_array = np.array([bbox[5] for bbox in bbox_list]).reshape(-1)
                
                scenario_agent_frame_info = {} # 序列中的一帧
                scenario_agent_frame_info['scenario_meta'] = meta_info # 场景元数据
                scenario_agent_frame_info['scenario_type'] = scenario_type # 场景类型
                scenario_agent_frame_info['vehicle_name'] = agent # 协同对象名称
                scenario_agent_frame_info['scene_name'] = '_'.join([scenario_type,scenario.split('_')[0],scenario.split('_')[-1]]) # 场景名称
                scenario_agent_frame_info['lidar_prefix'] = '_'.join([scenario_type,agent,frame]) # 雷达帧标识符
                scenario_agent_frame_info['lidar_path'] = osp.join(osp.abspath(data_path),scenario_type,agent,'lidar01',scenario,frame+contents['lidar01']) # 雷达路径
                scenario_agent_frame_info['bev_path'] = osp.join(osp.abspath(data_path),scenario_type,agent,'BEV_instance_camera',scenario,frame+contents['BEV_instance_camera']) # BEV实例路径
                scenario_agent_frame_info['timestamp'] = int(frame.split('_')[-1]) # 帧时间戳，帧间隔为0.1s
                scenario_agent_frame_info['scenario_length'] = last_seq_idx # 最大时间戳
                scenario_agent_frame_info['cams'] = dict()# 多相机
                scenario_agent_frame_info['cams'].update(cams) # 多个相机，每个相机包括图像路径，标定参数
                scenario_agent_frame_info['lidar_to_ego_matrix'] = calib_dict['lidar_to_ego'] # 雷达外参
                scenario_agent_frame_info['ego_to_world_matrix'] = calib_dict['ego_to_world'] # 对象的世界坐标
                scenario_agent_frame_info['vehicle_speed_x'] = agent_speed_x # 对象的x速度
                scenario_agent_frame_info['vehicle_speed_y'] = agent_speed_y # 对象的y速度
                scenario_agent_frame_info['gt_names'] = label_array # 世界类别gt
                scenario_agent_frame_info['gt_boxes'] = bbox_array # 世界boxes
                scenario_agent_frame_info['gt_velocity'] = vel_array# 世界速度gt
                scenario_agent_frame_info['vehicle_id'] = vehicle_id_array # 世界类别id
                scenario_agent_frame_info['num_lidar_pts'] = num_lidar_pts_array # 世界点云数目（目标上的点数）
                scenario_agent_frame_info['camera_visibility'] = camera_visibility_array # 世界目标是否可见
                scenario_agent_frame_info['sample_interval'] = sample_interval # 采样频率，决定帧间隔，原文为0.5s也就是间隔5采样

                result.append(scenario_agent_frame_info)
    return result


def create_deepaccident_info_file(data_path, pkl_prefix='deepaccident', save_path=None, sample_interval = 5):
    assert sample_interval >= 1
    dirs = os.listdir(data_path)
    total = len(deepaccident_folder_struct)
    for dir in dirs:
        if dir in deepaccident_folder_struct:
            total -= 1
    assert (total == 0),f"Dataset is not well formatted and requires the following structure: {deepaccident_folder_struct}"
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
    deepaccident_infos_train = _get_detail_info(train_split_items,data_path,sample_interval)
    filename = osp.join(save_path,f'{pkl_prefix}_infos_train.pkl')
    output_dict['data_list'] = deepaccident_infos_train
    mmengine.dump(output_dict, filename)
    mmengine.print_log(f'DeepAccident info train file is saved to {filename}','current')
    deepaccident_infos_val = _get_detail_info(val_split_items,data_path,sample_interval)
    filename = osp.join(save_path,f'{pkl_prefix}_infos_val.pkl')
    output_dict['data_list'] = deepaccident_infos_val
    mmengine.dump(output_dict, filename)
    mmengine.print_log(f'DeepAccident info val file is saved to {filename}','current')
    deepaccident_infos_test = _get_detail_info(test_split_items,data_path,sample_interval)
    filename = osp.join(save_path,f'{pkl_prefix}_infos_test.pkl')
    output_dict['data_list'] = deepaccident_infos_test
    mmengine.dump(output_dict, filename)
    mmengine.print_log(f'DeepAccident info test file is saved to {filename}, assume test equals val','current')

    
    