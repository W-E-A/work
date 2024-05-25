custom_imports = dict(
    imports=['projects.MyProject',],
    allow_failed_imports=False
)

# full with multi sweeps
train_annfile_path = '/ai/volume/work/data/deepaccident_ms/deepaccident_infos_train.pkl'
val_annfile_path = '/ai/volume/work/data/deepaccident_ms/deepaccident_infos_val.pkl'

# full no sweeps
# train_annfile_path = '/mnt/auto-labeling/wyc/wea_work/deepaccident/data/deepaccident/deepaccident_infos_train.pkl'
# val_annfile_path = '/mnt/auto-labeling/wyc/wea_work/deepaccident/data/deepaccident/deepaccident_infos_val.pkl'

# debug with multi sweeps
# train_annfile_path = '/mnt/auto-labeling/wyc/wea_work/deepaccident/data/deepaccident_ms_debug/deepaccident_infos_train.pkl'
# val_annfile_path = '/mnt/auto-labeling/wyc/wea_work/deepaccident/data/deepaccident_ms_debug/deepaccident_infos_val.pkl'

# debug no sweeps
# train_annfile_path = '/mnt/auto-labeling/wyc/wea_work/deepaccident/data/deepaccident_debug/deepaccident_infos_train.pkl'
# val_annfile_path = '/mnt/auto-labeling/wyc/wea_work/deepaccident/data/deepaccident_debug/deepaccident_infos_val.pkl'

use_multi_sweeps = True
delete_pointcloud = True # CLOUD

classes = [
    'car', 'van', 'truck', 'cyclist', 'motorcycle', 'pedestrian'
]
agents = [
    'ego_vehicle', 'other_vehicle', 'ego_vehicle_behind', 'other_vehicle_behind', 'infrastructure'
]

lidar_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
mask_range = [-3.0, -1.5, -5.0, 3.0, 1.5, 3.0]
det_center_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
motion_range = [-50, -50, -5.0, 50, 50, 3.0]
voxel_size = [0.1, 0.1, 8.0]
det_out_factor = 4
corr_out_factor = 4
motion_out_factor = 5
det_voxel_size = [voxel_size[0] * det_out_factor, voxel_size[1] * det_out_factor, voxel_size[2]]
corr_voxel_size = [voxel_size[0] * corr_out_factor, voxel_size[1] * corr_out_factor, voxel_size[2]]
motion_voxel_size = [voxel_size[0] * motion_out_factor, voxel_size[1] * motion_out_factor, voxel_size[2]]

det_with_velocity = True
code_size = 9
# code_size = 7
code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
# code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
det_tasks = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=2, class_names=['van', 'truck']),
    dict(num_class=2, class_names=['cyclist', 'motorcycle']),
    dict(num_class=1, class_names=['pedestrian']),
]
det_common_heads = dict(
    reg=(2, 2),
    height=(1, 2),
    dim=(3, 2),
    rot=(2, 2),
    vel=(2, 2),
)

batch_size = 2 # CLOUD
num_workers = 4 # CLOUD
seq_length = 8
present_idx = 2
sample_key_interval = 1
# sample_agents = (
#     'ego_vehicle', 'infrastructure',
# )
# sample_agents = (
#     'ego_vehicle', 'other_vehicle', 'infrastructure',
# )
sample_agents = tuple(agents)
infrastructure_name = 'infrastructure'
ego_name = 'ego_vehicle'
motion_only_vehicle = False
motion_filter_invalid = False
corr_only_vehicle = False
corr_filter_invalid = False
vehicle_id_list = [0, 1, 2] # agents 'car', 'van', 'truck'

train_pipline = [
    dict(
        type='LoadPointsNPZ',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,),
    dict(
        type='LoadPointsFromMultiSweepsNPZ',
        use_multi_sweeps=use_multi_sweeps,
        # remove_point_cloud_range=mask_range,
        remove_point_cloud_range=None,
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
    ),
    dict(
        type='LoadAnnotations3DV2X',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_3d_isvalid=True,
        with_track_id=True,
        ),
    dict(type='ConstructEGOBox', infrastructure_name=infrastructure_name),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    # dict( # FIXME how to deal with the feture wrapper
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.3925, 0.3925],
    #     scale_ratio_range=[0.95, 1.05],
    #     translation_std=[0, 0, 0]),
    # dict(
    #     type='RandomFlip3D',
    #     sync_2d=False,
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=lidar_range),
    dict(type='ObjectRangeFilterV2X', point_cloud_range=lidar_range),
    dict(type='ObjectNameFilterV2X', classes=classes),
    dict(type='ObjectTrackIDFilter', ids=[-1, ], impl=True),
    dict(type='ObjectValidFilter', impl=False),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputsV2X',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'bbox_3d_isvalid', 'track_id']
        # keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'track_id'] # validfilter impl
    ),
]

test_pipline = [
    dict(
        type='LoadPointsNPZ',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,),
    dict(
        type='LoadPointsFromMultiSweepsNPZ',
        use_multi_sweeps=use_multi_sweeps,
        # remove_point_cloud_range=mask_range,
        remove_point_cloud_range=None,
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
    ),
    dict(
        type='LoadAnnotations3DV2X',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_3d_isvalid=True,
        with_track_id=True,
        ),
    dict(type='ConstructEGOBox', infrastructure_name=infrastructure_name),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    dict(type='PointsRangeFilter', point_cloud_range=lidar_range),
    dict(type='ObjectRangeFilterV2X', point_cloud_range=lidar_range),
    dict(type='ObjectNameFilterV2X', classes=classes),
    dict(type='ObjectTrackIDFilter', ids=[-1, ], impl=True),
    dict(type='ObjectValidFilter', impl=False),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputsV2X',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'bbox_3d_isvalid', 'track_id']
        # keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'track_id'] # validfilter impl
    ),
]

train_scene_pipline = [
    dict(type='GatherV2XPoseInfo'),
    dict(
        type = 'CorrelationFilter',
        pc_range = lidar_range,
        voxel_size = voxel_size,
        infrastructure_name = infrastructure_name,
        with_velocity = det_with_velocity,
        ego_id = -100,
        min_distance_thres = 5,
        max_distance_thres = 20,
        alpha_coeff = 1,
        beta_coeff = 1,
        gamma_coeff = 2,
        enable_visualize=False,
        verbose = False,
    ),
    # dict(
    #     type = 'MakeMotionLabels',
    #     pc_range_motion = motion_range,
    #     voxel_size_motion = motion_voxel_size,
    #     pc_range_lidar = lidar_range,
    #     voxel_size_lidar = corr_voxel_size,
    #     infrastructure_name = infrastructure_name,
    #     just_present = False,
    #     ego_id = -100,
    #     motion_only_vehicle = motion_only_vehicle,
    #     corr_only_vehicle = corr_only_vehicle,
    #     motion_filter_invalid = motion_filter_invalid,
    #     corr_filter_invalid = corr_filter_invalid,
    #     vehicle_id_list = vehicle_id_list,
    #     ignore_index = 255,
    # ),
    # dict(type='DestoryEGOBox', ego_id = -100),
    dict(type='RemoveHistoryLabels'),
    dict(type='RemoveFutureLabels'),
    dict(type='RemoveHistoryInputs'),
    dict(type='RemoveFutureInputs'),
    dict(type='PackSceneInfo'),
    dict(type='DropSceneKeys',keys=('seq', 'sample_interval')),
]

test_scene_pipline = [
    dict(type='GatherV2XPoseInfo'),
    dict(
        type = 'CorrelationFilter',
        pc_range = lidar_range,
        voxel_size = voxel_size,
        infrastructure_name = infrastructure_name,
        with_velocity = det_with_velocity,
        ego_id = -100,
        min_distance_thres = 5,
        max_distance_thres = 20,
        alpha_coeff = 1,
        beta_coeff = 1,
        gamma_coeff = 2,
        enable_visualize=False,
        verbose = False,
    ),
    # dict(
    #     type = 'MakeMotionLabels',
    #     pc_range_motion = motion_range,
    #     voxel_size_motion = motion_voxel_size,
    #     pc_range_lidar = lidar_range,
    #     voxel_size_lidar = corr_voxel_size,
    #     infrastructure_name = infrastructure_name,
    #     just_present = False,
    #     ego_id = -100,
    #     motion_only_vehicle = motion_only_vehicle,
    #     corr_only_vehicle = corr_only_vehicle,
    #     motion_filter_invalid = motion_filter_invalid,
    #     corr_filter_invalid = corr_filter_invalid,
    #     vehicle_id_list = vehicle_id_list,
    #     ignore_index = 255,
    # ),
    # dict(type='DestoryEGOBox', ego_id = -100),
    dict(type='RemoveHistoryLabels'),
    dict(type='RemoveFutureLabels'),
    dict(type='RemoveHistoryInputs'),
    dict(type='RemoveFutureInputs'),
    dict(type='PackSceneInfo'),
    dict(type='DropSceneKeys',keys=('seq', 'sample_interval')),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
    sampler=dict(
          type='DefaultSampler',
          shuffle=True), # CLOUD
    dataset=dict(
        type = 'DeepAccident_V2X_Dataset',
        ann_file = train_annfile_path,
        pipeline = train_pipline,
        modality = dict(use_lidar=True, use_camera=False),
        box_type_3d = 'LiDAR',
        load_type = 'frame_based',
        key_interval = sample_key_interval,
        seq_length = seq_length,
        present_idx = present_idx,
        co_agents = sample_agents,
        filter_empty_gt = True,
        test_mode = False,
        scene_shuffle = False,
        with_velocity = det_with_velocity,
        adeptive_seq_length = True,
        scene_pipline = train_scene_pipline,
    ),
)

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(
          type='DefaultSampler',
          shuffle=False),
    dataset=dict(
        type = 'DeepAccident_V2X_Dataset',
        ann_file = val_annfile_path,
        pipeline = test_pipline,
        modality = dict(use_lidar=True, use_camera=False),
        box_type_3d = 'LiDAR',
        load_type = 'frame_based',
        key_interval = sample_key_interval,
        seq_length = seq_length,
        present_idx = present_idx,
        co_agents = sample_agents,
        filter_empty_gt = True,
        test_mode = True,
        scene_shuffle = False,
        with_velocity = det_with_velocity,
        adeptive_seq_length = True,
        scene_pipline = test_scene_pipline,
    ),
)

test_evaluator = dict(
    type='KittiMetricModified',
    metric=['iou_mAP',],
    with_velocity=det_with_velocity,
)

model = dict(
    type='EgoModel',
    corr_model = None,
    data_preprocessor=dict(
        type='DeepAccidentDataPreprocessor',
        delete_pointcloud=delete_pointcloud,
        voxel=True,
        voxel_type = 'hard',
        voxel_layer=dict(
            point_cloud_range=lidar_range,
            max_num_points=20,
            voxel_size=voxel_size,
            max_voxels=(30000, 40000)),
    ), # train, test voxel/pillar size
    pts_voxel_encoder = dict(
        type = 'PillarFeatureNet',
        in_channels = 5 if use_multi_sweeps else 4,
        feat_channels = (64, ),
        with_distance = False,
        with_cluster_center = True,
        with_voxel_center = True,
        voxel_size = voxel_size,
        point_cloud_range = tuple(lidar_range),
        norm_cfg = dict(
            type = 'BN1d',
            eps = 1e-3,
            momentum = 0.01),
        mode = 'max',
        legacy = False
    ),
    pts_middle_encoder=dict(
        type = 'PointPillarsScatterWrapper',
        in_channels = 64,
        lidar_range = lidar_range,
        voxel_size = voxel_size
    ),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True
    ),
    multi_task_head=dict(
        type='MTHead',
        det_head=dict(
            type='CenterHeadModified',
            in_channels=sum([128, 128, 128]),
            tasks=det_tasks,
            bbox_coder=dict(
                type='CenterPointBBoxCoder',
                post_center_range=det_center_range,
                max_num=500,
                score_threshold=0.1,
                out_size_factor=det_out_factor,
                voxel_size=voxel_size[:2],
                pc_range=lidar_range[:2],
                code_size=code_size),
            common_heads=det_common_heads,
            loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
            loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
            separate_head=dict(
                type='SeparateHead',
                head_conv=64,
                init_bias=-2.19,
                final_kernel=3
            ),
            share_conv_channel=64,
            num_heatmap_convs=2,
            norm_bbox=True,
            with_velocity=det_with_velocity,
        ),
    ),
    pts_train_cfg=dict(
        voxel_size=voxel_size,
        point_cloud_range=lidar_range,
        out_size_factor=det_out_factor,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=code_weights, # code_size
    ),
    pts_test_cfg=dict(
        nms_type='rotate',
        post_center_limit_range=det_center_range,
        score_threshold=0.1,
        nms_thr=[0.1, 0.1, 0.3, 0.3],
        nms_rescale_factor=[1.0, [0.7, 0.7], [2.0, 2.0], 4.5],
        pre_max_size=1000,
        post_max_size=83,
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 10, 12, 1, 0.85, 0.175], # FIXME circle nms
    ),
    pts_fusion_cfg=dict(
        corr_thresh = 0.3,
        train_ego_name=ego_name, # FIXME
        test_ego_name=ego_name,
        corr_pc_range=lidar_range,
    ),
    co_cfg=dict(
        infrastructure_name=infrastructure_name
    )
)



lr = 1 * 1e-4
checkpoint_interval = 2
max_checkpoint_num = 4
log_interval = 1

log_level = 'INFO'
load_from = None
resume = False

default_scope = 'mmdet3d'

default_hooks = dict(
                runtime_info=dict(type='RuntimeInfoHook'),
                timer=dict(type='IterTimerHook'),
                sampler_seed=dict(type='DistSamplerSeedHook'),
                logger=dict(type='LoggerHook', interval=log_interval),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(type='CheckpointHook', interval=checkpoint_interval, max_keep_ckpts= max_checkpoint_num),
            )
custom_hooks = [
    dict(type='ShowGPUMessage', interval=2, log_level='INFO', log_dir='/mnt/infra_dataset_ssd/ad_infra_dataset_pilot_fusion/checkpoints/gpu_messages')
] # CLOUD

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(
    window_size=10,
    by_epoch=True,
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))
# learning rate
param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from 0 to lr * 10
    # during the next 12 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=lr * 10,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=12,
        eta_min=lr * 1e-4,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=12,
        eta_min=1,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=20,
    # val_interval=1
)
# val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SimpleLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    pc_range=lidar_range,
    voxel_size=voxel_size,
    # mask_range=mask_range # FIXME
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).

auto_scale_lr = dict(enable=False, base_batch_size=32)

# dist
find_unused_parameters = True