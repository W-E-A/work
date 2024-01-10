_base_ = ['./exp1.py']

train_annfile_path = 'data/deepaccident/deepaccident_infos_train.pkl'
val_annfile_path = 'data/deepaccident/deepaccident_infos_val.pkl'

classes = [
    'car', 'van', 'truck', 'cyclist', 'motorcycle', 'pedestrian'
]

lidar_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# motion_range = [-50, -50, -5.0, 50, 50, 3.0]
motion_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 8.0]

# seq_length = 100
seq_length = 6
present_idx = 0
key_interval = 1
co_agents = ('ego_vehicle', 'infrastructure')
# co_agents = ('ego_vehicle',)
det_with_velocity = False
# code_size = 9
code_size = 7
# code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

det_common_heads = dict(
    reg=(2, 2),
    height=(1, 2),
    dim=(3, 2),
    rot=(2, 2),
)

train_batch_size = 1
# train_batch_size = 2
train_num_workers = 1

gpus = 4

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
    dict(type='ConstructEGOBox'),
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
    dict(type='ObjectValidIDFilterV2X', ids=[-100, -1], impl=False),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputsV2X',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'bbox_3d_isvalid', 'track_id']
        # keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'track_id']
    ),
]

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=train_num_workers,
    dataset=dict(
        # ann_file = train_annfile_path, # FIXME
        ann_file = val_annfile_path, # FIXME
        pipeline = train_pipline,
        key_interval = key_interval,
        seq_length = seq_length,
        present_idx = present_idx,
        co_agents = co_agents,
        with_velocity = det_with_velocity,
    ),
)

model = dict(
    multi_task_head=dict(
        det_head=dict(
            bbox_coder=dict(
                code_size=code_size
                ),
            common_heads=det_common_heads,
            with_velocity=det_with_velocity, # Modified
        ),
    ),
    pts_train_cfg=dict(
        code_weights=code_weights,
        gather_task_loss=True,
    ),
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=-1),
)

lr = 1e-4

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=20,
    # val_interval=1
)

# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).

auto_scale_lr = dict(enable=True, base_batch_size=train_batch_size * gpus)

# dist
find_unused_parameters = True
