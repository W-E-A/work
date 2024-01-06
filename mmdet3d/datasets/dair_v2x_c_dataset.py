from typing import Callable, List, Optional, Sequence, Union,  Dict, Tuple, Mapping, Any
import mmengine
from mmengine.config import Config
from mmengine.dataset import BaseDataset
from mmcv.transforms.base import BaseTransform
from mmengine.registry import FUNCTIONS
from mmdet3d.registry import DATASETS,TRANSFORMS
from .utils import tfm_to_pose,gen_pose_noise,pose_to_tfm,calc_relative_pose,corners_to_boxes,mask_gt_outside_range,boxes_to_corners,corner_to_standup_box
from .pcd_utils import read_pcd,shuffle_points,mask_ego_points,mask_points_by_range,downsample_lidar_minimum,lidar_project_einsum
from mmdet3d.structures.ops.box_overlaps import bbox_overlaps
import torch
import numpy as np
import math
from copy import deepcopy

@TRANSFORMS.register_module()
class AddPoseNoise(BaseTransform):
    """
    modify: lidar_to_world_matrix

    calib_dict + lidar_to_world_matrix_clean
    """
    def __init__(self,
                 impl = True,
                 pos_std = 0,
                 rot_std = 0,
                 pos_mean = 0,
                 rot_mean = 0
                 ) -> None:
        super().__init__()
        self.impl = impl
        self.pos_std = pos_std
        self.rot_std = rot_std
        self.pos_mean = pos_mean
        self.rot_mean = rot_mean

    # @profile
    def transform(self, results: Dict) -> Union[Dict,Tuple[List, List],None]:
        frame = results['scenario_frame']

        if self.impl and any([self.pos_std,self.rot_std,self.pos_mean,self.rot_mean]):
            for v in frame:
                lidar_pose = tfm_to_pose(v['calib_dict']['lidar_to_world_matrix'])
                lidar_pose_noise = lidar_pose + gen_pose_noise(self.pos_std,self.rot_std,self.pos_mean,self.rot_mean)
                v['calib_dict']['lidar_to_world_matrix_clean'] = v['calib_dict']['lidar_to_world_matrix']
                v['calib_dict']['lidar_to_world_matrix'] = pose_to_tfm(lidar_pose_noise)
        return results


@TRANSFORMS.register_module()
class SpecifiyEgo(BaseTransform):
    """
    results + ego_id + pose_list /+ pose_list_clean

    pose_list `all` to ego matrix
    """
    def __init__(self, ego_name = None, gen_clean = False) -> None:
        super().__init__()
        self.ego_name = ego_name
        self.gen_clean = gen_clean
    # @profile
    def transform(self, results: Dict) -> Union[Dict,Tuple[List, List],None]:
        frame = results['scenario_frame']
        agents = results['agents']

        ego_id = 0
        if self.ego_name is not None:
            assert self.ego_name in agents,"Invalid ego name"
            ego_id = agents.index(self.ego_name)
        results['ego_id'] = ego_id

        pose_list = calc_relative_pose(
            frame[ego_id]['calib_dict']['lidar_to_world_matrix'], 
            [item['calib_dict']['lidar_to_world_matrix'] for item in frame])
        results['pose_list'] = pose_list
        if self.gen_clean and 'lidar_to_world_matrix_clean' in frame[ego_id]['calib_dict'].keys():
            pose_list_clean = calc_relative_pose(
            frame[ego_id]['calib_dict']['lidar_to_world_matrix_clean'], 
            [item['calib_dict']['lidar_to_world_matrix_clean'] for item in frame])
            results['pose_list_clean'] = pose_list_clean
        return results


@TRANSFORMS.register_module()
class PreparePCD(BaseTransform):
    """
    TODO clean project's feature

    + pc + vis_pc

    """
    def __init__(self, 
                 lidar_range = [-100.8, -40, -3, 100.8, 40, 1], 
                 mask_ego_range = [-1.95, -1.1, 2.95, 1.1], 
                 visualize = False, 
                 project_to_ego = False) -> None:
        super().__init__()
        self.lidar_range = lidar_range
        self.mask_ego_range = mask_ego_range
        self.visualize = visualize
        self.project_to_ego = project_to_ego
    # @profile
    def transform(self, results: Dict) -> Union[Dict,Tuple[List, List],None]:
        frame = results['scenario_frame']
        for idx,item in enumerate(frame):
            pc, _ = read_pcd(item['lidar_path'])
            pc = shuffle_points(pc)
            pc = mask_ego_points(pc, self.mask_ego_range)
            if 'pose_list' in results.keys():
                prj_pc = lidar_project_einsum(pc[:,:3],results['pose_list'][idx])
                if self.project_to_ego:
                    pc[:,:3] = prj_pc
                if self.visualize:
                    if self.project_to_ego:
                        item['vis_pc'] = prj_pc
                    else:
                        item['vis_pc'] = pc[:,:3]
            pc = mask_points_by_range(pc,self.lidar_range)
            item['pc'] = pc
        
        if self.visualize:
            pc_list = [item['vis_pc'] for item in frame]
            down_pc_list = downsample_lidar_minimum(pc_list)
            for idx,item in enumerate(frame):
                item['vis_pc'] = down_pc_list[idx]

        return results


@TRANSFORMS.register_module()
class Voxelize(BaseTransform):
    """

    + voxel_features voxel_coords voxel_num_points
    """
    def __init__(self, 
                 lidar_range = [-100.8, -40, -3, 100.8, 40, 1], 
                 voxel_size = [0.4, 0.4, 4], 
                 num_point_feature = 4,
                 max_num_voxels = 32000,
                 max_points_per_voxel = 32) -> None:
        super().__init__()
        self.lidar_range = lidar_range
        self.voxel_size = voxel_size
        self.num_point_feature = num_point_feature
        self.max_num_voxels = max_num_voxels
        self.max_points_per_voxel = max_points_per_voxel

        from spconv.pytorch.utils import PointToVoxel

        self.voxelizer = PointToVoxel(
            vsize_xyz=self.voxel_size,
            coors_range_xyz=self.lidar_range,
            num_point_features=self.num_point_feature,
            max_num_voxels=self.max_num_voxels,
            max_num_points_per_voxel=self.max_points_per_voxel
        )
    # @profile
    def transform(self, results: Dict) -> Union[Dict,Tuple[List, List],None]:
        frame = results['scenario_frame']
        for item in frame:
            if 'pc' in item.keys():
                pc_torch = torch.from_numpy(item['pc'])
                features, coordinates, num_points = self.voxelizer(pc_torch)
                item['voxel_features'] = features.numpy() # type: ignore
                item['voxel_coords'] = coordinates.numpy() # type: ignore
                item['voxel_num_points'] = num_points.numpy() # type: ignore
        
        return results


@TRANSFORMS.register_module()
class GenerateGT(BaseTransform):
    """
    Generate each GT under each agent coord
    Generate Fusion GT under EGO coord
    """
    def __init__(self,
                 order = 'hwl',
                 anchor_l = 4.5, 
                 anchor_w = 2,
                 anchor_h = 1.56,
                 anchor_r = [0, 90],
                 num_anchors = 2,
                 downsample_rate = 2,
                 max_num_anchors=100,
                 pos_thres = 0.6,
                 neg_thres = 0.45,
                 voxel_size = [0.4, 0.4, 4],
                 lidar_range = [-100.8, -40, -3, 100.8, 40, 1],
                 filter_range_adding = [0, 0, -1, 0, 0, 1]) -> None:
        super().__init__()
        self.order = order
        self.anchor_l = anchor_l
        self.anchor_w = anchor_w
        self.anchor_h = anchor_h
        self.anchor_r = anchor_r
        self.num_anchors = num_anchors
        self.downsample_rate = downsample_rate
        self.max_num_anchors = max_num_anchors
        self.pos_thres = pos_thres
        self.neg_thres = neg_thres
        self.voxel_size = voxel_size
        self.lidar_range = lidar_range

        assert len(self.anchor_r) == self.num_anchors

        if self.order == 'lwh':
            raise NotImplementedError("GT encoder only support hwl order")

        self.W = math.ceil((self.lidar_range[3] - self.lidar_range[0]) / self.voxel_size[0]) // self.downsample_rate
        self.H = math.ceil((self.lidar_range[4] - self.lidar_range[1]) / self.voxel_size[1]) // self.downsample_rate
        self.D = math.ceil((self.lidar_range[5] - self.lidar_range[2]) / self.voxel_size[2])
        
        x = np.linspace(self.lidar_range[0] + self.voxel_size[0] / 2 * self.downsample_rate, self.lidar_range[3] - self.voxel_size[0] / 2 * self.downsample_rate, self.W) # [W]
        y = np.linspace(self.lidar_range[1] + self.voxel_size[1] / 2 * self.downsample_rate, self.lidar_range[4] - self.voxel_size[1] / 2 * self.downsample_rate, self.H) # [H]

        cx, cy = np.meshgrid(x, y)
         # center
        cx = np.tile(cx[..., np.newaxis], self.num_anchors) # [H, W, 2]
        cy = np.tile(cy[..., np.newaxis], self.num_anchors) # [H, W, 2]
        cz = np.full_like(cx, -1.0) #TODO ??? # [H, W, 2]
        w = np.full_like(cx, self.anchor_w)
        l = np.full_like(cx, self.anchor_l)
        h = np.full_like(cx, self.anchor_h)

        r = np.zeros_like(cx)

        for idx, ar in enumerate(self.anchor_r):
            r[..., idx] = math.radians(ar)
        
        if self.order == 'hwl': # pointpillar
            anchor_box = np.stack([cx, cy, cz, h, w, l, r], axis=-1)

        elif self.order == 'lhw':
            anchor_box = np.stack([cx, cy, cz, l, h, w, r], axis=-1)
        else:
            raise ValueError(f"order must be 'lwh' or 'hwl', got {self.order}")
        
        self.anchor_box = anchor_box

        self.filter_range = [a + b for a, b in zip(filter_range_adding, self.lidar_range)]

    def create_labels(self, boxes, corners):

        anchor_box = deepcopy(self.anchor_box)

        pos_equal_one = np.zeros((self.H, self.W, self.num_anchors))
        neg_equal_one = np.zeros((self.H, self.W, self.num_anchors))
        targets = np.zeros((self.H, self.W, self.num_anchors * 7))

        anchors = anchor_box.reshape(-1, 7) # [N, 7]
        anchors_d = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2) # [N, ]

        anchor_8points = boxes_to_corners(anchors, self.order)
        anchor_standup = corner_to_standup_box(anchor_8points) # [N, 4]

        gt_standup = corner_to_standup_box(corners) # [n, 4]
    
        iou = bbox_overlaps(
            np.ascontiguousarray(anchor_standup).astype(np.float32),
            np.ascontiguousarray(gt_standup).astype(np.float32)
        ) # [N, n]
        # iou.T [n, N]
        id_highest = np.argmax(iou.T, axis=1) # [n, ]
        id_highest_gt = np.arange(iou.T.shape[0]) # [n, ]
        
        mask = iou.T[id_highest_gt, id_highest] > 0 # [n, ]
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask] # [u, ]

        id_pos, id_pos_gt = np.where(iou > self.pos_thres)
        id_neg = np.where(np.sum(iou < self.neg_thres,axis=1) == iou.shape[1])[0]
        
        id_pos = np.concatenate([id_pos, id_highest]) # get pos id
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt]) # get pos gt
        id_pos, mask = np.unique(id_pos, return_index=True)
        id_pos_gt = id_pos_gt[mask]
        id_neg.sort()

        # cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(
            id_pos, (self.H, self.W, self.num_anchors)) # 从长条有效索引中直接按照给定的空间大小按照顺序恢复
        pos_equal_one[index_x, index_y, index_z] = 1
        # print(index_x)
        # print(index_y)
        # print(index_z)

        # calculate the targets
        # 0，7 prd * d + ac = gt
        targets[index_x, index_y, np.array(index_z) * 7] = \
            (boxes[id_pos_gt, 0] - anchors[id_pos, 0]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 1] = \
            (boxes[id_pos_gt, 1] - anchors[id_pos, 1]) / anchors_d[
                id_pos]
        targets[index_x, index_y, np.array(index_z) * 7 + 2] = \
            (boxes[id_pos_gt, 2] - anchors[id_pos, 2]) / anchors[
                id_pos, 3]
        # 3, 10 exp(prd * h)
        targets[index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            boxes[id_pos_gt, 3] / anchors[id_pos, 3])
        targets[index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            boxes[id_pos_gt, 4] / anchors[id_pos, 4])
        targets[index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            boxes[id_pos_gt, 5] / anchors[id_pos, 5])
        # 6, 13 prd + ac = gt
        targets[index_x, index_y, np.array(index_z) * 7 + 6] = (
                boxes[id_pos_gt, 6] - anchors[id_pos, 6])
    
        index_x, index_y, index_z = np.unravel_index(
            id_neg, (self.H, self.W, self.num_anchors))
        neg_equal_one[index_x, index_y, index_z] = 1

        # to avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(
            id_highest, (self.H, self.W, self.num_anchors))
        neg_equal_one[index_x, index_y, index_z] = 0

        return anchor_box, pos_equal_one, neg_equal_one, targets 

    def create_align_boxes(self, boxes_or_corners):

        boxes_or_corners, valid_mask = mask_gt_outside_range(
            boxes_or_corners,
            self.filter_range,
            self.order,
            return_mask=True
        )

        max_proj_boxes = np.zeros((self.max_num_anchors, 7))
        max_mask = np.zeros(self.max_num_anchors)
        object_ids = list(np.where(valid_mask == True)[0])

        return_more = False
        if boxes_or_corners.shape[1] == 8:
            boxes = corners_to_boxes(boxes_or_corners, self.order)
            return_more = True
        else:
            boxes = boxes_or_corners

        assert boxes.shape[0] < self.max_num_anchors
        for i, gt_boxes in enumerate(boxes):
            max_proj_boxes[i] = gt_boxes
            max_mask[i] = 1
            max_mask = max_mask > 0

        if return_more:
            return boxes_or_corners, boxes, max_proj_boxes, max_mask, object_ids
        else:
            return boxes_or_corners, max_proj_boxes, max_mask, object_ids

    # @profile
    def transform(self, results: Dict) -> Union[Dict,Tuple[List, List],None]:
        frame = results['scenario_frame']

        for idx, agent_frame in enumerate(frame):

            # gt_names = agent_frame['lidar_gt_names'] # (N, )

            gt_boxes, results[f'{idx}_gt_boxes'], results[f'{idx}_gt_mask'], results[f'{idx}_gt_object_ids'] = self.create_align_boxes(agent_frame['lidar_gt_boxes']) # type: ignore

            gt_8points = boxes_to_corners(gt_boxes)

            _, results[f'{idx}_pos_equal_one'], _, results[f'{idx}_labels'] = self.create_labels(gt_boxes, gt_8points)


        ego_frame = frame[0]
        if 'ego_id' in results.keys():
            for item in frame:
                if item['vehicle_name'] == results['agents'][results['ego_id']]:
                    ego_frame = item
                    break
        # (40,)
        # (40, 7)
        # (40, 8, 3)
        # gt_names = ego_frame['gt_names']
        # gt_boxes = ego_frame['gt_boxes']
        gt_8points = ego_frame['gt_8points']
        if 'lidar_to_world_matrix_clean' in ego_frame['calib_dict'].keys():
            ego_pos_matrix_clean = ego_frame['calib_dict']['lidar_to_world_matrix_clean']
        else:
            ego_pos_matrix_clean = ego_frame['calib_dict']['lidar_to_world_matrix']

        n8_points = gt_8points.reshape(-1, 3)

        proj_8points = lidar_project_einsum(n8_points, np.linalg.inv(ego_pos_matrix_clean))
        
        proj_8points = proj_8points.reshape(-1, 8, 3)

        proj_8points, proj_boxes, results['gt_boxes'],results['gt_mask'], results['gt_object_ids'] = self.create_align_boxes(proj_8points) # type: ignore

        results['anchor_box'], results['pos_equal_one'], _, results['labels'] = self.create_labels(proj_boxes, proj_8points)

        return results
        

@TRANSFORMS.register_module()
class GatherData(BaseTransform):
    """

    """
    def __init__(self,) -> None:
        super().__init__()

    # @profile
    def transform(self, results: Dict) -> Union[Dict,Tuple[List, List],None]:
        frame = results['scenario_frame']
        
        voxel_features = []
        voxel_coords = []
        voxel_num_points = []
        lidar_to_world_matrix = []
        pc_list = []
        vis_pc_list = []
        
        for item in frame:
            voxel_features.append(item.pop('voxel_features'))
            voxel_coords.append(item.pop('voxel_coords'))
            voxel_num_points.append(item.pop('voxel_num_points'))
            lidar_to_world_matrix.append(item['calib_dict'].pop('lidar_to_world_matrix'))
            pc_list.append(item.pop('pc'))
            if 'vis_pc' in item:
                vis_pc_list.append(item.pop('vis_pc'))
            
        
        results['voxel_features'] = voxel_features
        results['voxel_coords'] = voxel_coords
        results['voxel_num_points'] = voxel_num_points
        results['lidar_to_world_matrix'] = np.stack(lidar_to_world_matrix, axis=0)
        results['pc_list'] = pc_list
        if len(vis_pc_list) > 0:
            results['vis_pc_list'] = vis_pc_list
        
        return results



@TRANSFORMS.register_module()
class DropFrameKeys(BaseTransform):
    """

    + voxel_features voxel_coords voxel_num_points
    """
    def __init__(self,names:list) -> None:
        super().__init__()
        self.names = names
    # @profile
    def transform(self, results: Dict) -> Union[Dict,Tuple[List, List],None]:
        frame = results['scenario_frame']
        for name in self.names:
            for item in frame:  
                if name in item.keys():
                    item.pop(name)
        return results
    

@TRANSFORMS.register_module()
class DropKeys(BaseTransform):
    """

    + voxel_features voxel_coords voxel_num_points
    """
    def __init__(self,names:list) -> None:
        super().__init__()
        self.names = names
    # @profile
    def transform(self, results: Dict) -> Union[Dict,Tuple[List, List],None]:
        for name in self.names:
            if name in results.keys():
                results.pop(name)
        return results


# @TRANSFORMS.register_module()
# class Voxelize(BaseTransform):
#     """

#     + voxel_features voxel_coords voxel_num_points
#     """
#     def __init__(self,) -> None:
#         super().__init__()

    
#     def transform(self, results: Dict) -> Union[Dict,Tuple[List, List],None]:

        
#         return results


@FUNCTIONS.register_module()
def dair_v2x_c_collect_train(data_batch: Sequence) -> Any:
    collect_dict = {}
    collect_dict['agents'] = [batch['agents'] for batch in data_batch]
    collect_dict['ego_ids'] = [batch['ego_id'] for batch in data_batch]

    collect_dict['pose_list'] = torch.vstack([torch.from_numpy(np.array(batch['pose_list'])).unsqueeze(0) for batch in data_batch])
    # collect_dict['anchor_box'] = torch.vstack([torch.from_numpy(batch['anchor_box']).unsqueeze(0) for batch in data_batch])

    # collect_dict['gt_boxes'] = torch.vstack([torch.from_numpy(batch['gt_boxes']).unsqueeze(0) for batch in data_batch])
    # collect_dict['gt_mask'] = torch.vstack([torch.from_numpy(batch['gt_mask']).unsqueeze(0) for batch in data_batch])
    # collect_dict['gt_object_ids'] = [batch['gt_object_ids'] for batch in data_batch]
    collect_dict['pos_equal_one'] = torch.vstack([torch.from_numpy(batch['pos_equal_one']).unsqueeze(0) for batch in data_batch])
    # collect_dict['neg_equal_one'] = torch.vstack([torch.from_numpy(batch['neg_equal_one']).unsqueeze(0) for batch in data_batch])
    collect_dict['labels'] = torch.vstack([torch.from_numpy(batch['labels']).unsqueeze(0) for batch in data_batch])

    agent_size = len(collect_dict['agents'][0])

    for idx in range(agent_size):
        # collect_dict[f'{idx}_gt_boxes'] = torch.vstack([torch.from_numpy(batch[f'{idx}_gt_boxes']).unsqueeze(0) for batch in data_batch])
        # collect_dict[f'{idx}_gt_mask'] = torch.vstack([torch.from_numpy(batch[f'{idx}_gt_mask']).unsqueeze(0) for batch in data_batch])
        # collect_dict[f'{idx}_gt_object_ids'] = [batch[f'{idx}_gt_object_ids'] for batch in data_batch]
        collect_dict[f'{idx}_pos_equal_one'] = torch.vstack([torch.from_numpy(batch[f'{idx}_pos_equal_one']).unsqueeze(0) for batch in data_batch])
        # collect_dict[f'{idx}_neg_equal_one'] = torch.vstack([torch.from_numpy(batch[f'{idx}_neg_equal_one']).unsqueeze(0) for batch in data_batch])
        collect_dict[f'{idx}_labels'] = torch.vstack([torch.from_numpy(batch[f'{idx}_labels']).unsqueeze(0) for batch in data_batch])

    collect_dict['voxel_features'] = []
    collect_dict['voxel_coords'] = []
    collect_dict['voxel_num_points'] = []
    collect_dict['lidar_to_world_matrix'] = torch.vstack([torch.from_numpy(batch['lidar_to_world_matrix']).unsqueeze(0) for batch in data_batch])

    agent_size = len(data_batch[0]['agents'])
    agent_voxel_info = [[[] for j in range(3)] for i in range(agent_size)]

    for b, batch in enumerate(data_batch):
        for idx in range(agent_size):
            agent_voxel_info[idx][0].append(batch['voxel_features'][idx])
            agent_voxel_info[idx][1].append(np.pad(batch['voxel_coords'][idx], ((0, 0),(1, 0)), mode='constant', constant_values=b))
            agent_voxel_info[idx][2].append(batch['voxel_num_points'][idx])

    for i in range(agent_size):
            collect_dict['voxel_features'].append(torch.from_numpy(np.vstack(agent_voxel_info[i][0])))
            collect_dict['voxel_coords'].append(torch.from_numpy(np.vstack(agent_voxel_info[i][1])))
            collect_dict['voxel_num_points'].append(torch.from_numpy(np.concatenate(agent_voxel_info[i][2], axis=0)))

    return collect_dict

@FUNCTIONS.register_module()
def dair_v2x_c_collect_test(data_batch: Sequence) -> Any:
    collect_dict = {}
    collect_dict['agents'] = [batch['agents'] for batch in data_batch]
    collect_dict['ego_ids'] = [batch['ego_id'] for batch in data_batch]

    collect_dict['pose_list'] = torch.vstack([torch.from_numpy(np.array(batch['pose_list'])).unsqueeze(0) for batch in data_batch])
    collect_dict['anchor_box'] = torch.vstack([torch.from_numpy(batch['anchor_box']).unsqueeze(0) for batch in data_batch])

    collect_dict['gt_boxes'] = torch.vstack([torch.from_numpy(batch['gt_boxes']).unsqueeze(0) for batch in data_batch])
    collect_dict['gt_mask'] = torch.vstack([torch.from_numpy(batch['gt_mask']).unsqueeze(0) for batch in data_batch])
    collect_dict['gt_object_ids'] = [batch['gt_object_ids'] for batch in data_batch]
    collect_dict['pos_equal_one'] = torch.vstack([torch.from_numpy(batch['pos_equal_one']).unsqueeze(0) for batch in data_batch])
    # collect_dict['neg_equal_one'] = torch.vstack([torch.from_numpy(batch['neg_equal_one']).unsqueeze(0) for batch in data_batch])
    collect_dict['labels'] = torch.vstack([torch.from_numpy(batch['labels']).unsqueeze(0) for batch in data_batch])

    agent_size = len(collect_dict['agents'][0])

    for idx in range(agent_size):
        collect_dict[f'{idx}_gt_boxes'] = torch.vstack([torch.from_numpy(batch[f'{idx}_gt_boxes']).unsqueeze(0) for batch in data_batch])
        collect_dict[f'{idx}_gt_mask'] = torch.vstack([torch.from_numpy(batch[f'{idx}_gt_mask']).unsqueeze(0) for batch in data_batch])
        collect_dict[f'{idx}_gt_object_ids'] = [batch[f'{idx}_gt_object_ids'] for batch in data_batch]
        collect_dict[f'{idx}_pos_equal_one'] = torch.vstack([torch.from_numpy(batch[f'{idx}_pos_equal_one']).unsqueeze(0) for batch in data_batch])
        # collect_dict[f'{idx}_neg_equal_one'] = torch.vstack([torch.from_numpy(batch[f'{idx}_neg_equal_one']).unsqueeze(0) for batch in data_batch])
        collect_dict[f'{idx}_labels'] = torch.vstack([torch.from_numpy(batch[f'{idx}_labels']).unsqueeze(0) for batch in data_batch])

    collect_dict['voxel_features'] = []
    collect_dict['voxel_coords'] = []
    collect_dict['voxel_num_points'] = []
    collect_dict['lidar_to_world_matrix'] = torch.vstack([torch.from_numpy(batch['lidar_to_world_matrix']).unsqueeze(0) for batch in data_batch])

    agent_size = len(data_batch[0]['agents'])
    agent_voxel_info = [[[] for j in range(3)] for i in range(agent_size)]

    for b, batch in enumerate(data_batch):
        for idx in range(agent_size):
            agent_voxel_info[idx][0].append(batch['voxel_features'][idx])
            agent_voxel_info[idx][1].append(np.pad(batch['voxel_coords'][idx], ((0, 0),(1, 0)), mode='constant', constant_values=b))
            agent_voxel_info[idx][2].append(batch['voxel_num_points'][idx])

    for i in range(agent_size):
            collect_dict['voxel_features'].append(torch.from_numpy(np.vstack(agent_voxel_info[i][0])))
            collect_dict['voxel_coords'].append(torch.from_numpy(np.vstack(agent_voxel_info[i][1])))
            collect_dict['voxel_num_points'].append(torch.from_numpy(np.concatenate(agent_voxel_info[i][2], axis=0)))

    collect_dict['pc_list'] = [batch['pc_list'] for batch in data_batch]
    if 'vis_pc_list' in data_batch[0]:
        collect_dict['vis_pc_list'] = [batch['vis_pc_list'] for batch in data_batch]

    return collect_dict


@DATASETS.register_module()
class DAIR_V2X_C_Dataset(BaseDataset):
    """
    metainfo: classes agents

    data_list: list of dict

        dict_keys([
            'scenario_id', 'vehicle_name', 'agent_id', 'gt_names', 
            'gt_boxes', 'gt_8points', 'system_error_offset', 'image_path', 
            'lidar_path', 'calib_dict', 'lidar_gt_names', 'lidar_gt_boxes', 
            'camera_gt_names', 'camera_gt_boxes'])
        <class 'str'>
        <class 'str'>
        <class 'str'>
        <class 'numpy.ndarray'>
        <class 'numpy.ndarray'>
        <class 'numpy.ndarray'>
        <class 'dict'>
        <class 'str'>
        <class 'str'>
        <class 'dict'>
        <class 'numpy.ndarray'>
        <class 'numpy.ndarray'>
        <class 'numpy.ndarray'>
        <class 'numpy.ndarray'>

    self.data_list : agents scenario_frame sample_idx

    """
    METAINFO = {
        'classes': ('Car', 'Truck', 'Van', 'Bus'),
        'agents': ('vehicle', 'infrastructure')
    }
    def __init__(self,
                 ann_file: str,
                 metainfo: Union[Mapping, Config, None] = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 **kwargs):

        super().__init__(ann_file,
                         metainfo,
                         data_prefix=dict(),
                         pipeline=pipeline, # type: ignore
                         test_mode=test_mode,
                         **kwargs)
        
    def load_data_list(self) -> List[dict]:
        """
        dict_keys(['agents', 'scenario_frame', 'sample_idx', 'ego_id', 
        'pose_list', 'gt_boxes', 'gt_mask', 'gt_object_ids', 'anchor_box', 
        'pos_equal_one', 'neg_equal_one', 'labels', 'voxel_features', 
        'voxel_coords', 'voxel_num_points', 'lidar_to_world_matrix'])
        """

        annotations = mmengine.load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                            'keys')
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        assert 'agents' in self._metainfo,"Key 'agents' must exist in the metainfo"
        agent_size = len(self._metainfo['agents'])
        assert len(raw_data_list) % agent_size == 0, "There are missing scenario frames"
        scenario_frame_size = len(raw_data_list) // agent_size
        
        data_list = []
        for i in range(scenario_frame_size):
            scenario_frame = []
            for j in range(agent_size):
                data_info = raw_data_list[i * agent_size + j]
                data_info = self.parse_data_info(data_info)
                if isinstance(data_info, dict):
                    assert data_info['vehicle_name'] == self._metainfo['agents'][j]
                    scenario_frame.append(data_info)
                else:
                    raise TypeError('data_info should be a dict or list of dict, '
                                    f'but got {type(data_info)}')
            data_list.append(dict(agents = self._metainfo['agents'], scenario_frame = scenario_frame))
        return data_list

    def parse_data_info(self, raw_data_info: dict):

        if raw_data_info['vehicle_name'] == 'vehicle':
            vehicle_calib_dict = raw_data_info['calib_dict']
            mat1 = vehicle_calib_dict['novatel_to_world_matrix']
            mat2 = vehicle_calib_dict['lidar_to_novatel_matrix']
            mat3 = mat1 @ mat2
            vehicle_calib_dict['lidar_to_world_matrix'] = mat3
        elif raw_data_info['vehicle_name'] == 'infrastructure':
            system_error_offset = raw_data_info['system_error_offset']
            infrastructure_calib_dict = raw_data_info['calib_dict']
            infrastructure_calib_dict['lidar_to_world_matrix'] = infrastructure_calib_dict['virtuallidar_to_world_matrix']
            infrastructure_calib_dict['lidar_to_world_matrix'][0,3] += system_error_offset['delta_x']
            infrastructure_calib_dict['lidar_to_world_matrix'][1,3] += system_error_offset['delta_y']

        return raw_data_info