# dataset settings
dataset_type = 'SARDetDataset'
data_root = 'data/SARDet_DOTA_800pix/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
angle_version = 'le90' 
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        rect_classes=[0,1,2,3,4,5],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'sardet09_train_txt/',
        img_prefix=data_root + 'JPEGImages/',
        pipeline=train_pipeline,
        version=angle_version),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'sardet09_test_txt/',
        img_prefix=data_root + 'JPEGImages/',
        pipeline=test_pipeline,
        version=angle_version),
    test=dict(
        type=dataset_type, 
        ann_file=data_root + 'sardet09_test_txt/',
        img_prefix=data_root + 'JPEGImages/',
        pipeline=test_pipeline,
        version=angle_version)) 