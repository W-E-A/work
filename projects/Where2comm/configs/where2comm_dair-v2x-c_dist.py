_base_ = ['./where2comm_dair-v2x-c.py']

model_wrapper_cfg = dict(
    type='Where2commDDP',
    find_unused_parameters=True
)