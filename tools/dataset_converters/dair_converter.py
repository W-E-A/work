import os
import os.path as osp

import mmengine
import numpy as np

dair_v2x_c_categories = ('Car', 'Truck', 'Van', 'Bus') # fusion label
dair_v2x_spd_categories = ('Truck', 'Pedestrian', 'Tricyclist', 'Motorcyclist', 'Car', 'Van', 'Cyclist', 'Barrowlist', 'Bus') # fusion label
dair_v2x_c_folder_struct = ('cooperative', 'infrastructure-side', 'vehicle-side', 'train.json', 'val.json') # new split from CoAlign https://github.com/yifanlu0227/CoAlign
dair_v2x_spd_folder_struct = ('cooperative', 'infrastructure-side', 'vehicle-side', 'maps')
agent_folders = ('vehicle-side', 'infrastructure-side')
agent_names = ('vehicle', 'infrastructure')
contents = {
    'calib':'.json',
    'image':'.jpg',
    'label':'.json',
    'velodyne':'.pcd',
}


def _check_dataset_folder(data_path, struct):
    dirs = os.listdir(data_path)
    total = len(struct)
    for dir in dirs:
        if dir in struct:
            total -= 1
    assert (total == 0),f"Dataset is not well formatted and requires the following structure: {struct}"

# @profile
def _get_detail_info(target_items, data_path):
    co_data = mmengine.load(osp.join(data_path, 'cooperative', 'data_info.json'))
    co_dict = {osp.basename(item['vehicle_image_path']).split('.')[0] : item for item in co_data}
    filtered_co_dict = {idx : co_dict[idx] for idx in target_items}
    mmengine.print_log(f"total len: {len(filtered_co_dict)}",'current')
    result = []
    for idx,item in mmengine.track_iter_progress(list(filtered_co_dict.items())):
        co_labels = mmengine.load(osp.join(data_path,item['cooperative_label_path']))

        bbox_list = []
        for label in co_labels:
            cls_label = label['type']
            bbox = [label['3d_location']['x'],label['3d_location']['y'],label['3d_location']['z'],
                    label['3d_dimensions']['h'],label['3d_dimensions']['w'],label['3d_dimensions']['l'],
                    label['rotation']]
            world_8_points = label['world_8_points']
            bbox_list.append((cls_label,bbox,world_8_points))
        
        label_array = np.array([bbox[0] for bbox in bbox_list]).reshape(-1)
        bbox_array = np.array([bbox[1] for bbox in bbox_list]).reshape(-1, 7)
        world_8p_array = np.array([bbox[2] for bbox in bbox_list]).reshape(-1,8,3)

        for agent in agent_folders:
            agnet_name = agent.split('-')[0]

            image_path = osp.join(data_path,item[agnet_name + '_image_path']) # type: ignore
            pointcloud_path = osp.join(data_path,item[agnet_name + '_pointcloud_path']) # type: ignore

            agent_id = osp.basename(image_path).split('.')[0]
            
            calib_dict = {}
            calib_path = osp.join(data_path,agent,'calib')
            calib_items = os.listdir(calib_path)
            for calib in calib_items:
                calib_dict[calib] = mmengine.load(osp.join(calib_path,calib,agent_id+contents['calib']))
                if calib.find('_to_') != -1:
                    if 'transform' in calib_dict[calib].keys():
                        temp_dict = calib_dict[calib]['transform']
                    else:
                        temp_dict = calib_dict[calib]
                    T = np.eye(4,dtype=np.float64)
                    T[:3,3] = np.array(temp_dict['translation']).reshape(-1)
                    T[:3,:3] = np.array(temp_dict['rotation'])
                    calib_dict.pop(calib)
                    calib_dict[calib + '_matrix'] = T

            try:
                lidar_labels = mmengine.load(osp.join(data_path,agent,'label','lidar',agent_id+contents['label']))
            except FileNotFoundError:
                lidar_labels = mmengine.load(osp.join(data_path,agent,'label','virtuallidar',agent_id+contents['label']))
            except:
                raise Exception('Should not reach here')

            bbox_list = []
            for label in lidar_labels:
                cls_label = label['type']
                bbox = [label['3d_location']['x'],label['3d_location']['y'],label['3d_location']['z'],
                    label['3d_dimensions']['h'],label['3d_dimensions']['w'],label['3d_dimensions']['l'],
                    label['rotation']]
                bbox_list.append((cls_label,bbox))
            
            lidar_label_array = np.array([bbox[0] for bbox in bbox_list]).reshape(-1)
            lidar_bbox_array = np.array([bbox[1] for bbox in bbox_list]).reshape(-1, 7)

            camera_labels = mmengine.load(osp.join(data_path,agent,'label','camera',agent_id+contents['label']))

            bbox_list = []
            for label in camera_labels:
                cls_label = label['type']
                bbox = [label['3d_location']['x'],label['3d_location']['y'],label['3d_location']['z'],
                    label['3d_dimensions']['h'],label['3d_dimensions']['w'],label['3d_dimensions']['l'],
                    label['rotation']]
                bbox_list.append((cls_label,bbox))
            
            camera_label_array = np.array([bbox[0] for bbox in bbox_list]).reshape(-1)
            camera_bbox_array = np.array([bbox[1] for bbox in bbox_list]).reshape(-1, 7)

            scenario_agent_info = {} # 这里就是一帧，没有序列
            scenario_agent_info['scenario_id'] = idx # 协同场景编号
            scenario_agent_info['vehicle_name'] = agnet_name # 场景下的协同对象名称
            scenario_agent_info['agent_id'] = agent_id # 协同对象id
            scenario_agent_info['gt_names'] = label_array # 世界坐标系下的感知类型
            scenario_agent_info['gt_boxes'] = bbox_array # 世界坐标系下的感知结果 xyzhwlr 格式
            scenario_agent_info['gt_8points'] = world_8p_array # 世界坐标系下的8点标签
            scenario_agent_info['system_error_offset'] = item['system_error_offset'] # 平移信息
            scenario_agent_info['image_path'] = image_path # 协同对象的图像路径
            scenario_agent_info['lidar_path'] = pointcloud_path # 协同对象的雷达路径
            scenario_agent_info['calib_dict'] = calib_dict # 该对象的的所有位置和标定信息，也就是内外参，其中内参不变，外参变为了T矩阵
            scenario_agent_info['lidar_gt_names'] = lidar_label_array # 雷达时间戳下的感知结果类别
            scenario_agent_info['lidar_gt_boxes'] = lidar_bbox_array # 雷达时间戳下的感知结果boxes
            scenario_agent_info['camera_gt_names'] = camera_label_array # 相机时间戳下的感知结果类别
            scenario_agent_info['camera_gt_boxes'] = camera_bbox_array # 相机时间戳下的感知结果boxes

            result.append(scenario_agent_info)
    return result


def create_dair_v2x_c_info_file(data_path, pkl_prefix='dair-v2x-c', save_path=None):
    _check_dataset_folder(data_path,dair_v2x_c_folder_struct)
    train_split_items = mmengine.load(osp.join(data_path,'train.json'))
    val_split_items = mmengine.load(osp.join(data_path,'val.json'))
    # assume test equals val
    test_split_items = mmengine.load(osp.join(data_path,'val.json'))
    
    if save_path is None:
        save_path = data_path

    output_dict = {}
    output_dict['metainfo'] = {
        'classes': dair_v2x_c_categories,
        'agents': agent_names
    }

    mmengine.print_log('Generate info. this may take several minutes.','current')
    dair_v2x_c_infos_train = _get_detail_info(train_split_items,data_path)
    filename = osp.join(save_path,f'{pkl_prefix}_infos_train.pkl')
    output_dict['data_list'] = dair_v2x_c_infos_train
    mmengine.dump(output_dict, filename)
    mmengine.print_log(f'DAIR-V2X-C info train file is saved to {filename}','current')
    dair_v2x_c_infos_val = _get_detail_info(val_split_items,data_path)
    filename = osp.join(save_path,f'{pkl_prefix}_infos_val.pkl')
    output_dict['data_list'] = dair_v2x_c_infos_val
    mmengine.dump(output_dict, filename)
    mmengine.print_log(f'DAIR-V2X-C info val file is saved to {filename}','current')
    dair_v2x_c_infos_test = _get_detail_info(test_split_items,data_path)
    filename = osp.join(save_path,f'{pkl_prefix}_infos_test.pkl')
    output_dict['data_list'] = dair_v2x_c_infos_test
    mmengine.dump(output_dict, filename)
    mmengine.print_log(f'DAIR-V2X-C info test file is saved to {filename}, assume test equals val','current')

