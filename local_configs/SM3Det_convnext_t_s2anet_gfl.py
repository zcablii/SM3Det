_base_ = [
    '../configs/_base_/datasets/SOI_Det.py', '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]

angle_version = 'le135'
gpu_number = 8
num_classes=26
source_ratio = [2,1,1]
model = dict(
    type='TriSourceOneOneDetector', 
    backbone=dict(
        type='ConvNeXt_moe_MultiInput',
        MoE_Block_inds = [[],[0,2],[i*2 for i in range(5)],[0,2]],
        datasets=None,
        num_experts= 8, 
        top_k= 2,
        arch='tiny',
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint='../data/pretrained/convnext-tiny.pth')),
    neck=dict(
        type='MultitaskFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        extra_level=1, 
        add_extra_convs='on_output',
        num_outs=5),
    sar_bbox_head=dict(
        type='GFLHead',
        num_classes=num_classes,
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
    rgb_fam_head=dict(
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
    rgb_align_cfgs=dict(
        type='AlignConv',
        kernel_size=3,
        channels=256,
        featmap_strides=[8, 16, 32, 64, 128]),
    rgb_odm_head=dict(
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
    rgb_train_cfg=dict(
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
    rgb_test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000),
    ifr_fam_head=dict(
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
    ifr_align_cfgs=dict(
        type='AlignConv',
        kernel_size=3,
        channels=256,
        featmap_strides=[8, 16, 32, 64, 128]),
    ifr_odm_head=dict(
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
    ifr_train_cfg=dict(
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
    ifr_test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001, #/8*gpu_number,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'sar_rpn_head': dict(lr_mult=1.),
            'sar_rpn_head': dict(lr_mult=1.),
            'rgb_rpn_head': dict(lr_mult=1.),
            'rgb_roi_head': dict(lr_mult=1.),
            'ifr_rpn_head': dict(lr_mult=1.),
            'ifr_roi_head': dict(lr_mult=1.),
            'backbone': dict(lr_mult=1.),
            'neck': dict(lr_mult=1.),
        })
    )


branch_field = ['sar', 'rgb', 'ifr']

img_size = 800

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
 
rgb_train_dataset = dict(version=angle_version)

ifred_train_dataset = dict(version=angle_version)

data = dict(
    val_2=dict(
        version=angle_version),
    val_3=dict(
        version=angle_version),
    test=dict(
        version=angle_version),
    )

    

total_images = 46260+25028 + 17990# 24358
gpus = 8
batch_size = sum(source_ratio)
# evaluation

evaluation = dict(interval=total_images//(batch_size*gpus), metric='bbox',classwise=True)
evaluation2 = dict(interval=total_images//(batch_size*gpus), metric='mAP') 
evaluation3 = dict(interval=total_images//(batch_size*gpus), metric='mAP') 

# learning policy
lr_config = dict(
    policy='dynamic',
    warmup='linear',
    extra_args = {'T':3, 'b':0.4, 'ema': 0.001, 'backbone_policy':'sigmoid_kl', 'head_policy':'normal'},
    reweight_losses={
    'sar_loss_cls':'sar_bbox_head','sar_loss_bbox':'sar_bbox_head','sar_loss_dfl':'sar_bbox_head',
    'rgb_fam.loss_cls':'rgb_fam_head', 'rgb_fam.loss_bbox':'rgb_fam_head', 'rgb_odm.loss_cls':'rgb_odm_head','rgb_odm.loss_bbox':'rgb_odm_head',
    'ifr_fam.loss_cls':'ifr_fam_head', 'ifr_fam.loss_bbox':'ifr_fam_head', 'ifr_odm.loss_cls':'ifr_odm_head','ifr_odm.loss_bbox':'ifr_odm_head',},
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[total_images//(batch_size*gpus)*8, total_images//(batch_size*gpus)*11])
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=total_images//(batch_size*gpus)*12)
checkpoint_config = dict(interval=total_images//(batch_size*gpus))


data = dict(
    samples_per_gpu=batch_size,
    train_dataloader = dict(multi_datasets=True,
        source_ratio=source_ratio)
)
find_unused_parameters = True