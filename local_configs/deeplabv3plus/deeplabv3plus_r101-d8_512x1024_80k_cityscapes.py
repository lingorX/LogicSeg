_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/cityscapes.py', 
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101), 
    test_cfg=dict(mode='whole', is_hiera=False))

