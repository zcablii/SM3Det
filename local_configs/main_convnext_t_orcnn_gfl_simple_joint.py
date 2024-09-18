dataset_type1 = 'SARDet_hbb_trisource'
dataset_type2 = 'SARDetDotaIFRedDataset'
dataset_type3 = 'SARDetDotaIFRedDataset'
data_root = '/defaultShare/pubdata/remote_sensing/tri_source_detection/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
branch_field = ['sar', 'rgb', 'ifr']
img_size = 800
angle_version = 'le90'
sar_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(800, 800)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(type='MultiBranch', branch_field=['sar', 'rgb', 'ifr'], sar=True)
]
rgb_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90'),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        rect_classes=[0, 1, 2, 3, 4, 5, 16, 18],
        version='le90'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(800, 800)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(type='MultiBranch', branch_field=['sar', 'rgb', 'ifr'], rgb=True)
]
ifred_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90'),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        rect_classes=[0, 1, 2, 3, 4, 5, 16, 18],
        version='le90'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(800, 800)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(type='MultiBranch', branch_field=['sar', 'rgb', 'ifr'], ifr=True)
]
sar_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect_subdataset', keys=['img'], subdataset='sar')
        ])
]
rgb_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect_subdataset', keys=['img'], subdataset='rgb')
        ])
]
ifred_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect_subdataset', keys=['img'], subdataset='ifr')
        ])
]
sar_train_dataset = dict(
    type='SARDet_hbb_trisource',
    ann_file=
    '/defaultShare/pubdata/remote_sensing/tri_source_detection/SARDet_50K/Annotations/after_merge_train.json',
    img_prefix=
    '/defaultShare/pubdata/remote_sensing/tri_source_detection/SARDet_50K/JPEGImages/',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(800, 800)),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        dict(type='MultiBranch', branch_field=['sar', 'rgb', 'ifr'], sar=True)
    ])
rgb_train_dataset = dict(
    type='SARDetDotaIFRedDataset',
    ann_file=
    '/defaultShare/pubdata/remote_sensing/tri_source_detection/DOTA_800pix/train/annfiles/',
    img_prefix=
    '/defaultShare/pubdata/remote_sensing/tri_source_detection/DOTA_800pix/train/images/',
    cache_annotations='cache/rgb_train_trisource/cache_annotations.pkl',
    cache_filtered='cache/rgb_train_trisource/cache_filtered.pkl',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RResize', img_scale=(800, 800)),
        dict(
            type='RRandomFlip',
            flip_ratio=[0.25, 0.25, 0.25],
            direction=['horizontal', 'vertical', 'diagonal'],
            version='le90'),
        dict(
            type='PolyRandomRotate',
            rotate_ratio=0.5,
            angles_range=180,
            auto_bound=False,
            rect_classes=[0, 1, 2, 3, 4, 5, 16, 18],
            version='le90'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(800, 800)),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        dict(type='MultiBranch', branch_field=['sar', 'rgb', 'ifr'], rgb=True)
    ],
    version='le90')
ifred_train_dataset = dict(
    type='SARDetDotaIFRedDataset',
    ann_file=
    '/defaultShare/pubdata/remote_sensing/tri_source_detection/DroneVehicle/dota_train/annfiles/',
    img_prefix=
    '/defaultShare/pubdata/remote_sensing/tri_source_detection/DroneVehicle/dota_train/png_images/',
    cache_annotations='cache/ifred_train_trisource/cache_annotations.pkl',
    cache_filtered='cache/ifred_train_trisource/cache_filtered.pkl',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RResize', img_scale=(800, 800)),
        dict(
            type='RRandomFlip',
            flip_ratio=[0.25, 0.25, 0.25],
            direction=['horizontal', 'vertical', 'diagonal'],
            version='le90'),
        dict(
            type='PolyRandomRotate',
            rotate_ratio=0.5,
            angles_range=180,
            auto_bound=False,
            rect_classes=[0, 1, 2, 3, 4, 5, 16, 18],
            version='le90'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(800, 800)),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        dict(type='MultiBranch', branch_field=['sar', 'rgb', 'ifr'], ifr=True)
    ],
    version='le90')
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='SARDet_hbb_trisource',
                ann_file=
                '/defaultShare/pubdata/remote_sensing/tri_source_detection/SARDet_50K/Annotations/after_merge_train.json',
                img_prefix=
                '/defaultShare/pubdata/remote_sensing/tri_source_detection/SARDet_50K/JPEGImages/',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size=(800, 800)),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes',
                                              'gt_labels']),
                    dict(
                        type='MultiBranch',
                        branch_field=['sar', 'rgb', 'ifr'],
                        sar=True)
                ]),
            dict(
                type='SARDetDotaIFRedDataset',
                ann_file=
                '/defaultShare/pubdata/remote_sensing/tri_source_detection/DOTA_800pix/train/annfiles/',
                img_prefix=
                '/defaultShare/pubdata/remote_sensing/tri_source_detection/DOTA_800pix/train/images/',
                cache_annotations=
                'cache/rgb_train_trisource/cache_annotations.pkl',
                cache_filtered='cache/rgb_train_trisource/cache_filtered.pkl',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='RResize', img_scale=(800, 800)),
                    dict(
                        type='RRandomFlip',
                        flip_ratio=[0.25, 0.25, 0.25],
                        direction=['horizontal', 'vertical', 'diagonal'],
                        version='le90'),
                    dict(
                        type='PolyRandomRotate',
                        rotate_ratio=0.5,
                        angles_range=180,
                        auto_bound=False,
                        rect_classes=[0, 1, 2, 3, 4, 5, 16, 18],
                        version='le90'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size=(800, 800)),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes',
                                              'gt_labels']),
                    dict(
                        type='MultiBranch',
                        branch_field=['sar', 'rgb', 'ifr'],
                        rgb=True)
                ],
                version='le90'),
            dict(
                type='SARDetDotaIFRedDataset',
                ann_file=
                '/defaultShare/pubdata/remote_sensing/tri_source_detection/DroneVehicle/dota_train/annfiles/',
                img_prefix=
                '/defaultShare/pubdata/remote_sensing/tri_source_detection/DroneVehicle/dota_train/png_images/',
                cache_annotations=
                'cache/ifred_train_trisource/cache_annotations.pkl',
                cache_filtered='cache/ifred_train_trisource/cache_filtered.pkl',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(type='RResize', img_scale=(800, 800)),
                    dict(
                        type='RRandomFlip',
                        flip_ratio=[0.25, 0.25, 0.25],
                        direction=['horizontal', 'vertical', 'diagonal'],
                        version='le90'),
                    dict(
                        type='PolyRandomRotate',
                        rotate_ratio=0.5,
                        angles_range=180,
                        auto_bound=False,
                        rect_classes=[0, 1, 2, 3, 4, 5, 16, 18],
                        version='le90'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size=(800, 800)),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes',
                                              'gt_labels']),
                    dict(
                        type='MultiBranch',
                        branch_field=['sar', 'rgb', 'ifr'],
                        ifr=True)
                ],
                version='le90')
        ]),
    val=dict(
        type='SARDet_hbb_trisource',
        ann_file=
        '/defaultShare/pubdata/remote_sensing/tri_source_detection/SARDet_50K/Annotations/after_merge_test.json',
        img_prefix=
        '/defaultShare/pubdata/remote_sensing/tri_source_detection/SARDet_50K/JPEGImages/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(
                        type='Collect_subdataset',
                        keys=['img'],
                        subdataset='sar')
                ])
        ]),
    val_2=dict(
        type='SARDetDotaIFRedDataset',
        ann_file=
        '/defaultShare/pubdata/remote_sensing/tri_source_detection/DOTA_800pix/val/annfiles/',
        img_prefix=
        '/defaultShare/pubdata/remote_sensing/tri_source_detection/DOTA_800pix/val/images/',
        cache_annotations='cache/rgb_val_trisource/cache_annotations.pkl',
        cache_filtered='cache/rgb_val_trisource/cache_filtered.pkl',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 800),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect_subdataset',
                        keys=['img'],
                        subdataset='rgb')
                ])
        ],
        version='le90'),
    val_3=dict(
        type='SARDetDotaIFRedDataset',
        ann_file=
        '/defaultShare/pubdata/remote_sensing/tri_source_detection/DroneVehicle/dota_test/annfiles/',
        img_prefix=
        '/defaultShare/pubdata/remote_sensing/tri_source_detection/DroneVehicle/dota_test/png_images/',
        cache_annotations='cache/ifred_val_trisource/cache_annotations.pkl',
        cache_filtered='cache/ifred_val_trisource/cache_filtered.pkl',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 800),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect_subdataset',
                        keys=['img'],
                        subdataset='ifr')
                ])
        ],
        version='le90'),
    test=dict(
        type='SARDetDotaIFRedDataset',
        ann_file=
        '/defaultShare/pubdata/remote_sensing/tri_source_detection/DroneVehicle/dota_test/annfiles/',
        img_prefix=
        '/defaultShare/pubdata/remote_sensing/tri_source_detection/DroneVehicle/dota_test/png_images/',
        cache_annotations='cache/ifred_val_trisource/cache_annotations.pkl',
        cache_filtered='cache/ifred_val_trisource/cache_filtered.pkl',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 800),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect_subdataset',
                        keys=['img'],
                        subdataset='ifr')
                ])
        ],
        version='le90'),
    train_dataloader=dict(multi_datasets=False, source_ratio=[1, 1, 1]),
    val_dataloader=dict(samples_per_gpu=2, workers_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=2, workers_per_gpu=2))
evaluation = dict(interval=1, metric='bbox', classwise=True)
optimizer = dict(
    type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=1000, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
gpu_number = 8
num_classes = 26
model = dict(
    type='TriSourceDetector',
    backbone=dict(
        type='ConvNeXt_moe_MultiInput',
        datasets=None,
        arch='tiny',
        drop_path_rate=0.1,
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint='../data/pretrained/convnext-tiny.pth')),
    neck=dict(
        type='MultitaskFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        extra_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    sar_bbox_head=dict(
        type='GFLHead',
        num_classes=26,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    rgb_rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version='le90',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range='le90',
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    rgb_roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=26,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range='le90',
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    ifr_rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version='le90',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range='le90',
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    ifr_roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=26,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range='le90',
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    sar_train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    sar_test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100),
    rgb_train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    rgb_test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000)),
    ifr_train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    ifr_test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000)))
find_unused_parameters = True
evaluation2 = dict(interval=1, metric='mAP')
work_dir = './work_dirs/main_convnext_t_orcnn_gfl_simple_joint'
auto_resume = True
gpu_ids = range(0, 1)
