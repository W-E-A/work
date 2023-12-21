_base_ = [
    './DeepAccident_singleframe_tiny.py'
]

# upper limit
# receptive_field = 5
# future_frames = 12
# # 16853MB
# receptive_field = 5
# future_frames = 6

# 12457MB # single-frame_tiny的感受野和未来预测帧的配置？
receptive_field = 3
future_frames = 4



# # 14259MB single gpu single batch size:  3 days, 14:04:41
# receptive_field = 5
# future_frames = 4

# # 15929MB single gpu single batch size: 4 days, 15:46:56
# receptive_field = 7
# future_frames = 4

# # 19223MB
# receptive_field = 9
# future_frames = 4

# # 22183MB single gpu single batch size: 6 days, 23:04:01
# receptive_field = 11
# future_frames = 4



# # 14287MB
# receptive_field = 3
# future_frames = 6

# # 17023MB
# receptive_field = 3
# future_frames = 8

# # 18791MB
# receptive_field = 3
# future_frames = 10

# # 21339MB single gpu single batch size: 3 days, 18:16:41
# receptive_field = 3
# future_frames = 12





future_discount = 0.95 # 特定的配置

model = dict(
    img_backbone=dict(
        pretrained='./data/swin_tiny_patch4_window7_224.pth',
    ),
    temporal_model=dict( # 修改了模型的temporal_model
        type='Temporal3DConvModel',
        receptive_field=receptive_field,
        input_egopose=True,
        in_channels=64,
        input_shape=(128, 128),
        with_skip_connect=True,
    ),
    pts_bbox_head=dict( # 修改了检测头的设置
        task_enbale={
            # '3dod': True, 'map': True, 'motion': True,
            '3dod': True, 'map': False, 'motion': True,
        },
        task_weights={
            '3dod': 0.5, 'map': 10.0, 'motion': 1.0,
        },
        cfg_motion=dict(
            type='IterativeFlow',
            task_dict={
                'segmentation': 2,
                'instance_center': 1,
                'instance_offset': 2,
                'instance_flow': 2,
            },
            receptive_field=receptive_field,
            n_future=future_frames,
            using_spatial_prob=True,
            n_gru_blocks=1,
            future_discount=future_discount,
            loss_weights={
                'loss_motion_seg': 5.0,
                'loss_motion_centerness': 1.0,
                'loss_motion_offset': 1.0,
                'loss_motion_flow': 1.0,
                'loss_motion_prob': 10.0,
            },
            sample_ignore_mode='past_valid',
            posterior_with_label=False,
        ),
    ),
    train_cfg=dict(
        pts=dict(
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    ),
)

data = dict( # 修改数据的配置
    # samples_per_gpu=2,
    # workers_per_gpu=4,
    samples_per_gpu=2,
    # workers_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        receptive_field=receptive_field,
        future_frames=future_frames,
    ),
    val=dict(
        receptive_field=receptive_field,
        future_frames=future_frames,
    ),
    test=dict(
        receptive_field=receptive_field,
        future_frames=future_frames,
    ),
)

# # get evaluation metrics every n epochs
# evaluation = dict(interval=1)
workflow = [('train', 1)] # 工作流 ？？？

# # fp16 settings, the loss scale is specifically tuned to avoid Nan
# fp16 = dict(loss_scale='dynamic')
optimizer = dict(type='AdamW', lr=2e-3, weight_decay=0.01) #设定优化器