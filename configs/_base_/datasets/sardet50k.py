# dataset settings
dataset_type = 'SARDet_hbb' 
data_root = 'data/SOI_Det/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
 

img_size = 800

angle_version = 'le90' 

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(img_size, img_size), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(img_size, img_size)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(img_size, img_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(img_size, img_size)),
            dict(type='ImageToTensor', keys=['img']),# dict(type='DefaultFormatBundle'), Not sure which of these
            dict(type='Collect', keys=['img'])
        ])
]
 

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'SARDet_50K/Annotations/train.json',
        img_prefix=data_root + 'SARDet_50K/JPEGImages/',
        pipeline=train_pipeline), 
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'SARDet_50K/Annotations/test.json',
        img_prefix=data_root + 'SARDet_50K/JPEGImages/',
        pipeline=test_pipeline,),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'SARDet_50K/Annotations/test.json',
        img_prefix=data_root + 'SARDet_50K/JPEGImages/',
        pipeline=test_pipeline),
    
)
    