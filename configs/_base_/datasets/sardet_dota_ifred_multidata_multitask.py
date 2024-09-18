# dataset settings
dataset_type1 = 'SARDet_hbb_trisource'
dataset_type2 = 'SARDetDotaIFRedDataset'
dataset_type3 = 'SARDetDotaIFRedDataset'
data_root = 'data/tri_source_detection/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Checklist: dataset_type 
# img_size
# rect_classes
# file_path

branch_field = ['sar', 'rgb', 'ifr']

img_size = 800

angle_version = 'le90' 

sar_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(img_size, img_size), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
 
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(img_size, img_size)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        sar=True)
]

rgb_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(img_size, img_size)),
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
        rect_classes=[0,1,2,3,4,5,16,18],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(img_size, img_size)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        rgb=True)
]

ifred_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(img_size, img_size)),
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
        rect_classes=[0,1,2,3,4,5,16,18],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(img_size, img_size)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(
        type='MultiBranch',
        branch_field=branch_field,
        ifr=True)
]
 

sar_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(img_size, img_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),# dict(type='DefaultFormatBundle'), Not sure which of these
            dict(type='Collect_subdataset', keys=['img'], subdataset='sar')
        ])
]


rgb_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(img_size, img_size),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect_subdataset', keys=['img'], subdataset='rgb')
        ])
]
 
ifred_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(img_size, img_size),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect_subdataset', keys=['img'], subdataset='ifr')
        ])
]

sar_train_dataset = dict(
        type=dataset_type1,
        ann_file=data_root + 'SARDet_50K/Annotations/after_merge_train.json',
        img_prefix=data_root + 'SARDet_50K/JPEGImages/',
        pipeline=sar_train_pipeline)


rgb_train_dataset = dict(
        type=dataset_type2,
        ann_file=data_root + 'DOTA_800pix/train/annfiles/',
        img_prefix=data_root + 'DOTA_800pix/train/images/',
        cache_annotations='cache/rgb_train_trisource/cache_annotations.pkl',
        cache_filtered='cache/rgb_train_trisource/cache_filtered.pkl',
        pipeline=rgb_train_pipeline, version=angle_version)

ifred_train_dataset = dict(
        type=dataset_type3,
        ann_file=data_root + 'DroneVehicle/dota_train/annfiles/',
        img_prefix=data_root + 'DroneVehicle/dota_train/png_images/',
        cache_annotations='cache/ifred_train_trisource/cache_annotations.pkl',
        cache_filtered='cache/ifred_train_trisource/cache_filtered.pkl',
        pipeline=ifred_train_pipeline, version=angle_version)


data = dict(
    samples_per_gpu=3,
    workers_per_gpu=4,
    train=dict(type='ConcatDataset', datasets=[sar_train_dataset, rgb_train_dataset, ifred_train_dataset]), 
    val=dict(
        type=dataset_type1,
        ann_file=data_root + 'SARDet_50K/Annotations/after_merge_test.json',
        img_prefix=data_root + 'SARDet_50K/JPEGImages/',
        pipeline=sar_test_pipeline,),
    val_2=dict(
        type=dataset_type2,
        ann_file=data_root + 'DOTA_800pix/val/annfiles/',
        img_prefix=data_root + 'DOTA_800pix/val/images/',
        cache_annotations='cache/rgb_val_trisource/cache_annotations.pkl',
        cache_filtered='cache/rgb_val_trisource/cache_filtered.pkl',
        pipeline=rgb_test_pipeline,
        version=angle_version),
    val_3=dict(
        type=dataset_type3,
        ann_file=data_root + 'DroneVehicle/dota_test/annfiles/',
        img_prefix=data_root + 'DroneVehicle/dota_test/png_images/',
        cache_annotations='cache/ifred_val_trisource/cache_annotations.pkl',
        cache_filtered='cache/ifred_val_trisource/cache_filtered.pkl',
        pipeline=ifred_test_pipeline,
        version=angle_version),
    test=dict(
        type=dataset_type3,
        ann_file=data_root + 'DroneVehicle/dota_test/annfiles/',
        img_prefix=data_root + 'DroneVehicle/dota_test/png_images/',
        cache_annotations='cache/ifred_val_trisource/cache_annotations.pkl',
        cache_filtered='cache/ifred_val_trisource/cache_filtered.pkl',
        pipeline=ifred_test_pipeline,
        version=angle_version),
  
    train_dataloader = dict(multi_datasets=True,
        source_ratio=[1, 1, 1]),
    val_dataloader = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,),
    test_dataloader = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,)
    )

    