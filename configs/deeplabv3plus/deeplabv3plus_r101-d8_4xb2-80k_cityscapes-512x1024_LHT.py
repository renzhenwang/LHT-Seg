
_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)                      

custom_imports = dict(
    imports=['mmseg.models.losses.label_hierarchy_transition_loss'],
    allow_failed_imports=False
)

model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(
        loss_decode=dict(
            type='LabelHierarchyTransitionLoss',
            loss_weight=1.0
        )
    ),
    data_preprocessor=data_preprocessor
)

randomness = dict(seed=0, diff_rank_seed=True)
