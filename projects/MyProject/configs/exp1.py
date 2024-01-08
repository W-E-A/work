custom_imports = dict(
    imports=['projects.MyProject.models',],
    allow_failed_imports=False
)

train_annfile_path = 'data/deepaccident/deepaccident_infos_train.pkl'
val_annfile_path = 'data/deepaccident/deepaccident_infos_val.pkl'
test_annfile_path = 'data/deepaccident/deepaccident_infos_val.pkl'

classes = [
    'car', 'van', 'truck', 'cyclist', 'motorcycle', 'pedestrian'
]

agents = [
    'ego_vehicle', 'other_vehicle', 'ego_vehicle_behind', 'other_vehicle_behind', 'infrastructure'
]

lidar_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
det_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
det_center_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
motion_range = [-50, -50, -5.0, 50, 50, 3.0]
voxel_size = [0.1, 0.1, 8.0]
grid_size = [1024, 1024, 1]
out_factor = 4
det_out_factor = 4
motion_out_factor = 4

seq_length = 100
present_idx = 1
# co_agents = ('ego_vehicle', 'infrastructure')
co_agents = ('ego_vehicle',)
det_with_velocity = True
det_tasks = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=1, class_names=['van']),
    dict(num_class=1, class_names=['truck']),
    dict(num_class=1, class_names=['cyclist']),
    dict(num_class=1, class_names=['motorcycle']),
    dict(num_class=1, class_names=['pedestrian']),
]
det_common_heads = dict(
    reg=(2, 2),
    height=(1, 2),
    dim=(3, 2),
    rot=(2, 2),
    vel=(2, 2)
)

train_batch_size = 1
train_num_workers = 1

train_pipline = [
    dict(
        type='LoadPointsNPZ',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,),
    dict(
        type='LoadAnnotations3DV2X',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_3d_isvalid=True,
        with_track_id=True,
        ),
    # dict(type='ObjectSample', db_sampler=db_sampler),
    # dict(
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
    dict(type='ObjectTrackIDFilterV2X', ids=[-100, -1]), # ego
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputsV2X',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'bbox_3d_isvalid', 'track_id'])
]

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=train_num_workers,
    pin_memory=True,
    drop_last=True,
    sampler=dict(
          type='DefaultSampler',
          shuffle=True),
    dataset=dict(
        type = 'DeepAccident_V2X_Dataset',
        ann_file = val_annfile_path, # TODO
        pipeline = train_pipline,
        modality = dict(use_lidar=True, use_camera=False),
        box_type_3d = 'LiDAR',
        load_type = 'frame_based',
        seq_length = seq_length,
        present_idx = present_idx,
        co_agents = co_agents,
        filter_empty_gt = True,
        test_mode = False,
        with_velocity = det_with_velocity,
        adeptive_seq_length = True,
        scene_pipline = [dict(type='PackSceneInfo'),
                         dict(type='DropSceneKeys',keys=('seq',)),],
    ),
)

model = dict(
    type='ProjectModel',
    data_preprocessor=dict(
        type='DeepAccidentDataPreprocessor',
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
        in_channels = 4,
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
    temporal_backbone=dict(
        type='TemporalIdentity',
        position='last'
    ),
    multi_task_head=dict(
        type='MTHead',
        det_head=dict(
            type='CenterHead',
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
                code_size=9),
            common_heads=det_common_heads,
            loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
            loss_bbox=dict(
                type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
            separate_head=dict(
                type='SeparateHead', init_bias=-2.19, final_kernel=3),
            share_conv_channel=64,
            num_heatmap_convs=2,
            norm_bbox=True
        ),
    ),
    pts_train_cfg=dict(
        grid_size=grid_size,
        voxel_size=voxel_size,
        point_cloud_range=lidar_range,
        out_size_factor=det_out_factor,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
    ),
    pts_test_cfg=dict(
        nms_type='rotate',
        post_center_limit_range=det_center_range,
        score_threshold=0.1,
        nms_thr=0.2,
        pre_max_size=1000,
        post_max_size=83,
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        out_size_factor=det_out_factor,
        voxel_size=voxel_size[:2],
        pc_range=lidar_range[:2]
    ),
)


default_scope = 'mmdet3d'

default_hooks = dict(
                runtime_info=dict(type='RuntimeInfoHook'),
                timer=dict(type='IterTimerHook'),
                sampler_seed=dict(type='DistSamplerSeedHook'),
                logger=dict(type='LoggerHook', interval=1),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(type='CheckpointHook', interval=-1),
            )

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(
    window_size=10,
    by_epoch=True,
)

log_level = 'INFO'
load_from = None
resume = False

lr = 1e-4

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
# test_cfg = dict(type='TestLoop')

visualizer=dict(
    type='Visualizer',
    # name='comm_vis',
    vis_backends=[
        # dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
    # save_dir='comm_mask'
)

# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).

# auto_scale_lr = dict(enable=False, base_batch_size=32)
