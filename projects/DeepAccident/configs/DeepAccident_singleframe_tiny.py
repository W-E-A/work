'''
single-frame BEVerse with swin-tiny backbone

单车的baseline 基于小型的swin-transformer骨干的BEVerse模型配置文件
'''

# 继承了所有的默认配置文件
_base_ = [
    '../../configs/_base_/datasets/nus-3d.py',
    '../../configs/_base_/schedules/poly_decay_20e.py',
    '../../configs/_base_/default_runtime.py'
]

# 23351MB for single-GPU training
find_unused_parameters = False
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
sync_bn = True

# ===== 重写默认的数据配置 ===== #

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# 修改有效点云范围（立体对角点）
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# point_cloud_range = [-76.0, -76.0, -5.0, 76.0, 76.0, 3.0]

# For carla we do 6-class detection
# class_names = [
#     'car', 'van', 'truck', 'cyclist', 'motorcycle', 'pedestrian'
# ]
# 检测类型
class_names = [
    'car', 'truck', 'van', 'cyclist', 'motorcycle', 'pedestrian',
    'invalid1', 'invalid2', 'invalid3', 'invalid4'
]

# Image-view augmentation 图像增强设置
data_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    # 'final_dim': (128, 352),
    'rot_lim': (-5.4, 5.4),
    'H': 900, 'W': 1600,
    'rand_flip': True,
    'bot_pct_lim': (0.0, 0.22),
    'crop_h': (0.0, 0.0),
    'cams': ['Camera_FrontLeft', 'Camera_Front', 'Camera_FrontRight',
             'Camera_BackLeft', 'Camera_Back', 'Camera_BackRight'],
    'Ncams': 6,
}

# bev数据增强设置
bev_aug_params = {
    'rot_range': [-0.3925, 0.3925],
    'scale_range': [0.95, 1.05],
    'trans_std': [0, 0, 0],
    'hflip': 0.5,
    'vflip': 0.5,
}

# det_grid_conf = {
#     'xbound': [-51.2, 51.2, 0.8],
#     'ybound': [-51.2, 51.2, 0.8],
#     'zbound': [-10.0, 10.0, 20.0],
#     'dbound': [1.0, 60.0, 1.0],
# }
#
# motion_grid_conf = {
#     'xbound': [-50.0, 50.0, 0.5],
#     'ybound': [-50.0, 50.0, 0.5],
#     'zbound': [-10.0, 10.0, 20.0],
#     'dbound': [1.0, 60.0, 1.0],
# }

# 检测网格置信度
det_grid_conf = {
    'xbound': [-51.2, 51.2, 0.8],
    'ybound': [-51.2, 51.2, 0.8],
    'zbound': [-30.0, 30.0, 60.0],
    'dbound': [1.0, 60.0, 1.0],
}

# 运动预测置信度
motion_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-30.0, 30.0, 60.0],
    'dbound': [1.0, 60.0, 1.0],
}

# det_grid_conf = {
#     'xbound': [-76.0, 76.0, 0.5],
#     'ybound': [-76.0, 76.0, 0.5],
#     'zbound': [-30.0, 30.0, 60.0],
#     'dbound': [1.0, 60.0, 1.0],
# }
#
# motion_grid_conf = {
#     'xbound': [-76.0, 76.0, 0.5],
#     'ybound': [-76.0, 76.0, 0.5],
#     'zbound': [-30.0, 30.0, 60.0],
#     'dbound': [1.0, 60.0, 1.0],
# }

# 地图网格置信度
map_grid_conf = {
    'xbound': [-30.0, 30.0, 0.15],
    'ybound': [-15.0, 15.0, 0.15],
    'zbound': [-30.0, 30.0, 60.0],
    'dbound': [1.0, 60.0, 1.0],
}

# 网格置信度
grid_conf = det_grid_conf

receptive_field = 1 # 感受野大小？？？
future_frames = 0 # 预测未来的帧数
future_discount = 1.0 # ？？？

voxel_size = [0.1, 0.1, 0.2] # 网格大小
model = dict(
    type='BEVerse',
    img_backbone=dict(
        type='SwinTransformer',
        # pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
        pretrained='./data/DeepAccident_data/swin_tiny_patch4_window7_224.pth',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(2, 3,),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.0,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        pretrain_style='official',
        output_missing_index_as_none=False),
    img_neck=dict(
        type='FPN_LSS', in_channels=384+768, inverse=True,),
    transformer=dict(
        type='TransformerLSS',
        grid_conf=grid_conf,
        input_dim=data_aug_conf['final_dim'],
        numC_input=512,
        numC_Trans=64,
    ),
    temporal_model=dict( # ？？？
        type='TemporalIdentity',
        grid_conf=grid_conf,
    ),
    pts_bbox_head=dict( # 多任务检测头
        type='MultiTaskHead',
        in_channels=64,
        # in_channels=320,
        out_channels=256,
        grid_conf=grid_conf,
        det_grid_conf=det_grid_conf,
        map_grid_conf=map_grid_conf,
        motion_grid_conf=motion_grid_conf,
        using_ego=True,
        task_enbale={
            '3dod': True,
            'map': True,
            'motion': True,
        },
        task_weights={
            '3dod': 1.0,
            'map': 10.0,
            'motion': 1.0,
        },
        bev_encode_block='Basic',
        cfg_3dod=dict(
            type='CenterHeadv1',
            in_channels=256,
            # tasks=[
            #     dict(num_class=1, class_names=['car']),
            #     dict(num_class=2, class_names=['van', 'truck']),
            #     dict(num_class=2, class_names=['cyclist', 'motorcycle']),
            #     dict(num_class=1, class_names=['pedestrian']),
            # ],

            tasks=[
                dict(num_class=1, class_names=['car']),
                dict(num_class=2, class_names=[
                    'truck', 'van']),
                dict(num_class=2, class_names=['invalid1', 'invalid2']),
                dict(num_class=1, class_names=['invalid3']),
                dict(num_class=2, class_names=['motorcycle', 'cyclist']),
                dict(num_class=2, class_names=['pedestrian', 'invalid4']),
            ],
            common_heads=dict(reg=(2, 2), height=(
                1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
            share_conv_channel=64,
            bbox_coder=dict(
                type='CenterPointBBoxCoder',
                pc_range=point_cloud_range[:2],
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=500,
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                code_size=9),
            separate_head=dict(
                type='SeparateHead', init_bias=-2.19, final_kernel=3),
            loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
            loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
            norm_bbox=True,
        ),
        cfg_map=dict(
            type='MapHead',
            task_dict={
                'semantic_seg': 4,
            },
            in_channels=256,
            class_weights=[1.0, 2.0, 2.0, 2.0],
            semantic_thresh=0.25,
        ),
        cfg_motion=dict(
            type='MotionHead',
            task_dict={
                'segmentation': 2,
                'instance_center': 1,
                'instance_offset': 2,
                'instance_flow': 2,
            },
            in_channels=256,
            grid_conf=motion_grid_conf,
            class_weights=[1.0, 2.0],
            receptive_field=receptive_field,
            n_future=future_frames,
            future_discount=future_discount,
            using_focal_loss=True,
            prob_latent_dim=32,
            future_dim=6,
            distribution_log_sigmas=[-5.0, 5.0],
            n_gru_blocks=3,
            n_res_layers=3,
            loss_weights={
                'loss_motion_seg': 1.0,
                'loss_motion_centerness': 1.0,
                'loss_motion_offset': 1.0,
                'loss_motion_flow': 1.0,
                'loss_motion_prob': 100,
            },
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            # grid_size=[2432, 2432, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=['rotate', 'rotate', 'rotate',
                      'circle', 'rotate', 'rotate'],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[1.0, [0.7, 0.7], [
                0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]],
        )))

# ===== 数据集配置 ===== #

dataset_type = 'DeepAccidentDataset' # 数据类型
data_root = 'data/DeepAccident_data/' # 原始数据根目录
data_info_path = 'data/carla_infos/' # 数据标签根目录

# 训练数据处理管道
train_pipeline = [
    # load image and apply image-view augmentation
    dict(type='LoadMultiViewImageFromFiles_DeepAccident', using_ego=True, temporal_consist=True,
         is_train=True, data_aug_conf=data_aug_conf),
    # load 3D bounding boxes & bev-semantic-maps
    dict(type='LoadAnnotations3D_MTL', with_bbox_3d=True,
         with_label_3d=True, with_instance_tokens=True),
    # bev-augmentations
    dict(
        type='MTLGlobalRotScaleTrans',
        rot_range=bev_aug_params['rot_range'],
        scale_ratio_range=bev_aug_params['scale_range'],
        translation_std=bev_aug_params['trans_std'],
        update_img2lidar=True),
    dict(
        type='MTLRandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=bev_aug_params['hflip'],
        flip_ratio_bev_vertical=bev_aug_params['vflip'],
        update_img2lidar=True),
    # # convert map labels
    # dict(type='RasterizeMapVectors', map_grid_conf=map_grid_conf),
    # convert motion labels
    dict(type='ConvertMotionLabels_DeepAccident', grid_conf=motion_grid_conf, only_vehicle=False),
    # filter objects
    dict(type='ObjectValidFilter'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    # bundle & collect
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D',
         # ['sample_idx', 'pts_filename', 'sweeps', 'timestamp', 'data_root', 'future_egomotions', 'has_invalid_frame',
         # 'img_is_valid', 'img_info', 'lidar2ego_rots', 'lidar2ego_trans', 'ann_info', 'vectors', 'img_fields',
         # 'bbox3d_fields', 'pts_mask_fields', 'pts_seg_fields', 'bbox_fields', 'mask_fields', 'seg_fields',
         # 'box_type_3d', 'box_mode_3d', 'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'instance_tokens',
         # 'gt_valid_flag', 'gt_vis_tokens', 'transformation_3d_flow', 'pcd_rotation', 'pcd_scale_factor', 'pcd_trans',
         # 'extrinsics', 'aug_transform', 'flip', 'flip_direction', 'pcd_horizontal_flip', 'pcd_vertical_flip',
         # 'semantic_map', 'instance_map', 'semantic_indices', 'forward_direction', 'backward_direction',
         # 'motion_segmentation', 'motion_instance', 'instance_centerness', 'instance_offset', 'instance_flow']
         keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'future_egomotions', 'aug_transform', 'img_is_valid',
               'motion_segmentation', 'motion_instance', 'instance_centerness', 'instance_offset', 'instance_flow',
               'has_invalid_frame'],
         meta_keys=('scenario_length', 'timestamp', 'filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape', 'lidar2ego_rots', 'lidar2ego_trans',
                    'lidar_to_ego_matrix', 'ego_to_world_matrix',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'img_info')),
]

# 测试数据处理管道
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_DeepAccident',
         using_ego=True, data_aug_conf=data_aug_conf),
    dict(type='LoadAnnotations3D_MTL', with_bbox_3d=True,
         with_label_3d=True, with_instance_tokens=True),
    # dict(type='RasterizeMapVectors', map_grid_conf=map_grid_conf),
    dict(type='ConvertMotionLabels_DeepAccident', grid_conf=motion_grid_conf, only_vehicle=False),

    dict(type='ObjectValidFilter'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'future_egomotions', 'gt_bboxes_3d',
                      'gt_labels_3d', 'has_invalid_frame', 'img_is_valid',
                      'motion_segmentation','motion_instance', 'instance_centerness', 'instance_offset', 'instance_flow'
                      ],
                meta_keys=('scenario_length', 'timestamp', 'filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                           'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                           'pcd_scale_factor', 'pcd_rotation', 'pts_filename', 'transformation_3d_flow', 'img_info', 'lidar2ego_rots', 'lidar2ego_trans',
                           'lidar_to_ego_matrix', 'ego_to_world_matrix', 'accident_visibility',
                           'scenario_type', 'town_name', 'weather', 'time_of_the_day', 'collision_status', 'junction_type', 'trajectory_type')),
        ],
    ),
]

# 输入模态，用于改变数据加载的行为
input_modality = dict(
    use_camera=True, # RGB
    use_lidar=False,
    use_radar=False,
    use_map=False,
    use_external=False,
    prototype='lift-splat-shoot', # 输入格式遵循论文LSS的设置
)

# 数据配置
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_info_path + 'carla_infos_train.pkl', # ext
        pipeline=train_pipeline, # ext
        classes=class_names,
        test_mode=False,
        receptive_field=receptive_field,
        future_frames=future_frames,
        grid_conf=grid_conf,
        map_grid_conf=map_grid_conf,
        modality=input_modality,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        receptive_field=receptive_field,
        future_frames=future_frames,
        grid_conf=grid_conf,
        map_grid_conf=map_grid_conf,
        ann_file=data_info_path + 'carla_infos_val.pkl',
        modality=input_modality),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        receptive_field=receptive_field,
        future_frames=future_frames,
        grid_conf=grid_conf,
        map_grid_conf=map_grid_conf,
        ann_file=data_info_path + 'carla_infos_val.pkl',
        modality=input_modality),
)

# ===== 优化配置 ===== #

# optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
optimizer = dict(type='AdamW', lr=2e-3, weight_decay=0.01)
evaluation = dict(interval=999, pipeline=test_pipeline)
