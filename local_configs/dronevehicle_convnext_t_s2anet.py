_base_ = [
    '../configs/_base_/datasets/dronevehicle.py', '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]
 
gpu_number = 8
# fp16 = dict(loss_scale='dynamic') 
num_classes=5
angle_version = 'le135'
model = dict(
    type='S2ANet',
    backbone=dict(
        type='ConvNeXt_moe_MultiInput',
        MoE_Block_inds = [[],[],[],[]],
        datasets=None,
        arch='tiny',
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint='../data/pretrained/convnext-tiny.pth')),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    fam_head=dict(
        type='RotatedRetinaHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            scales=[4],
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    align_cfgs=dict(
        type='AlignConv',
        kernel_size=3,
        channels=256,
        featmap_strides=[8, 16, 32, 64, 128]),
    odm_head=dict(
        type='ODMRefineHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='PseudoAnchorGenerator', strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    train_cfg=dict(
        fam_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        odm_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        version=angle_version),
    val=dict(
        version=angle_version),
    test=dict(
        version=angle_version)) 

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001, #/8*gpu_number,
    betas=(0.9, 0.999),
    weight_decay=0.05)

log_config = dict(interval=200)
find_unused_parameters = True
