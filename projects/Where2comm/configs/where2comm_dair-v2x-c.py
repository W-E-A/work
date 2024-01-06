custom_imports = dict(
    imports=['projects.Where2comm', 'projects.Where2comm.models'],
    allow_failed_imports=False
)

lidar_range = [-100.8, -40, -3, 100.8, 40, 1]
voxel_size = [0.4, 0.4, 4]
num_anchors = 2
order = 'hwl'
downsample_rate = 2

train_annfile_path = '/ai/volume/work/data/dair/dair-v2x-c_infos_train.pkl' # TODO
val_annfile_path = '/ai/volume/work/data/dair/dair-v2x-c_infos_val.pkl'
test_annfile_path = '/ai/volume/work/data/dair/dair-v2x-c_infos_test.pkl'

train_pipline = [
     dict(type = 'AddPoseNoise',
          impl = False,
          pos_std = 0.2,
          rot_std = 0.2,
          pos_mean = 0,
          rot_mean = 0),
     dict(type = 'SpecifiyEgo',
          ego_name = 'vehicle',
          gen_clean = False),
     dict(type = 'PreparePCD',
          lidar_range = lidar_range, 
          mask_ego_range = [-1.95, -1.1, 2.95, 1.1], 
          visualize = False, 
          project_to_ego = False),
     dict(type = 'Voxelize',
          lidar_range = lidar_range, 
          voxel_size = voxel_size, 
          num_point_feature = 4,
          max_num_voxels = 32000,
          max_points_per_voxel = 32),
     dict(type = 'GenerateGT',
          order = order,
          anchor_l = 4.5, 
          anchor_w = 2,
          anchor_h = 1.56,
          anchor_r = [0, 90],
          num_anchors = num_anchors,
          downsample_rate = downsample_rate,
          max_num_anchors=100,
          pos_thres = 0.6,
          neg_thres = 0.45,
          voxel_size = voxel_size,
          lidar_range = lidar_range,
          filter_range_adding = [0, 0, -1, 0, 0, 1]),
     dict(type = 'GatherData'),
     dict(type = 'DropFrameKeys',
          names = [
              'agent_id',
              'gt_names',
              'gt_boxes',
              'gt_8points',
              'system_error_offset',
              'image_path',
              'lidar_gt_names',
              'lidar_gt_boxes',
              'camera_gt_names',
              'camera_gt_boxes'
          ]
     )
]

test_pipline = [
     dict(type = 'SpecifiyEgo',
          ego_name = 'vehicle',
          gen_clean = False),
     dict(type = 'PreparePCD',
          lidar_range = lidar_range, 
          mask_ego_range = [-1.95, -1.1, 2.95, 1.1], 
          visualize = True, 
          project_to_ego = False),
     dict(type = 'Voxelize',
          lidar_range = lidar_range, 
          voxel_size = voxel_size, 
          num_point_feature = 4,
          max_num_voxels = 70000,
          max_points_per_voxel = 32),
     dict(type = 'GenerateGT',
          order = order,
          anchor_l = 4.5, 
          anchor_w = 2,
          anchor_h = 1.56,
          anchor_r = [0, 90],
          num_anchors = num_anchors,
          downsample_rate = downsample_rate,
          max_num_anchors=100,
          pos_thres = 0.6,
          neg_thres = 0.45,
          voxel_size = voxel_size,
          lidar_range = lidar_range,
          filter_range_adding = [0, 0, -1, 0, 0, 1]),
     dict(type = 'GatherData'),
     dict(type = 'DropFrameKeys',
          names = [
              'agent_id',
              'gt_names',
              'gt_boxes',
              'gt_8points',
              'system_error_offset',
              'image_path',
              'lidar_gt_names',
              'lidar_gt_boxes',
              'camera_gt_names',
              'camera_gt_boxes'
          ]
     )
]

train_dataloader = dict(
    batch_size=10,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    sampler=dict(
          type='DefaultSampler',
          shuffle=True),
    dataset=dict(
         type = 'DAIR_V2X_C_Dataset',
         ann_file = train_annfile_path,
         pipeline = train_pipline,
         test_mode = False
    ),
    collate_fn=dict(type='dair_v2x_c_collect_train')
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    pin_memory=True,
    drop_last=False,
    sampler=dict(
          type='DefaultSampler',
          shuffle=False),
    dataset=dict(
         type = 'DAIR_V2X_C_Dataset',
         ann_file = test_annfile_path,
         pipeline = test_pipline,
         test_mode = True
    ),
    collate_fn=dict(type='dair_v2x_c_collect_test')
)

test_evaluator = dict(
    type='V2XMetric'
)

model = dict(
    type = 'Where2comm',
    co_agents = 2,
    voxel_size = voxel_size,
    downsample_rate = downsample_rate,
    data_preprocessor = dict(type='BaseDataPreprocessor'),
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
        legacy = True
    ),
    pts_middle_encoder=dict(
        type = 'PointPillarsScatterWrapper',
        in_channels = 64,
        lidar_range = lidar_range,
        voxel_size = voxel_size
    ),
    pts_backbone=dict(
        type = 'BEVBackbone',
        layers = [3, 4, 5],
        layer_strides = [2, 2, 2],
        num_filters = [64, 128, 256],
        upsample_strides = [1, 2, 4],
        num_upsample_filters = [128, 128, 128],
    ),
    pts_shrink_module=dict(
        type = 'ShrinkModule',
        in_channels = 3 * 128,
        out_channels = 256,
        shrink_step = 1,
        norm = False
    ),
#     pts_compress_module=dict(
#         type = CompressModule,
#         in_channels = 256,
#         compress_ratio = 2
#     ),
    pts_detect_module=dict(
        type = 'DetectHead',
        in_channels = 256,
        num_anchors = num_anchors
    ),
    pts_comm_module=dict(
        type = 'Communication',
        thres = 0.01,
        smooth = True,
        smooth_k_size = 7,
        smooth_sigma = 1.0
    ),
    pts_fusion_module=dict(
        type = 'ScaledDotProductAttenFusion'
    ),
    # pts_loss_module=dict(
    #     type = 'Where2commLoss',
    #     num_classes = 2,
    #     cls_weight = 1.0,
    #     cls_alpha = 0.25,
    #     focal_gamma = 2.0,
    #     reg_weight = 2.0,
    #     use_dir = False,
    #     smooth_beta = 1.0 / 9.0
    # ),
    pts_loss_module=dict(
        type = 'PointPillarLoss',
    ),
    test_cfg=dict(
        agent_threshold = [0.01, 0.01],
        fusion_threshold = 0.01,
        order = order,
        nms_threshold = 0.15,
        lidar_range = lidar_range,
        only_vis = True
    )
)

default_scope = 'mmdet3d'

default_hooks = dict(
                runtime_info=dict(type='RuntimeInfoHook'),
                timer=dict(type='IterTimerHook'),
                sampler_seed=dict(type='DistSamplerSeedHook'),
                logger=dict(type='LoggerHook', interval=1),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(type='CheckpointHook', interval=2),
            )

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(
    window_size=10,
    by_epoch=True,
    # custom_cfg=[
    #     dict(
    #         data_src='ego_comm_rate',
    #         method_name='current',
    #         window_size=1
    #     ),
    #     dict(
    #         data_src='infra_comm_rate',
    #         method_name='current',
    #         window_size=1
    #     ),
    # ]
)

log_level = 'INFO'
load_from = None
resume = False

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.002,
        weight_decay=1e-4
     ),
    # max_norm=10 is better for SECOND
#     clip_grad=dict(max_norm=35, norm_type=2)
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=30,
    # val_interval=1
)
# val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=2.0 / 1000,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=30,
        by_epoch=True,
        milestones=[20, 27],
        gamma=0.1)
]

visualizer=dict(
    type='Visualizer',
    # name='comm_vis',
    vis_backends=[
        # dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
    # save_dir='comm_mask'
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).

# auto_scale_lr = dict(enable=False, base_batch_size=32)