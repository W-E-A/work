_base_ = ['./where2comm_dair-v2x-c.py']

model = dict(
    train_cfg = dict(
        fix_cfg = dict(
            fix_voxel_encoder = False,
            fix_middle_encoder = False,
            fix_shared_backbone = False,
            fix_shrink_module = False,
            fix_compress_encoder = False,
            fix_comm_module = False,
            fix_fusion = False,
            fix_fusion_head = False,
            fix_detect_head = False,
        )
    ),
    init_cfg = dict(
        type='Pretrained',
        checkpoint='work_dirs/where2comm_dair-v2x-c_raw/epoch_30.pth'
    ),
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0002,
        weight_decay=1e-4
     ),
    # max_norm=10 is better for SECOND
#     clip_grad=dict(max_norm=35, norm_type=2)
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=20,
    # val_interval=1
)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=2.0 / 10000,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[13, 18],
        gamma=0.5)
]