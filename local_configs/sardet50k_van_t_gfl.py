_base_ = [
    '../configs/_base_/datasets/sardet50k.py', '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]
 
gpu_number = 8
# fp16 = dict(loss_scale='dynamic') 
num_classes=6
model = dict(
    type='GFL',
    backbone=dict(
        type='VAN',
        embed_dims=[32, 64, 160, 256],
        drop_rate=0.1,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        init_cfg=dict(type='Pretrained', checkpoint="../data/pretrained/van_tiny_754.pth.tar"),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    neck=dict(
        type='FPN',
        in_channels=[32, 64, 160, 256],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
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
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))


find_unused_parameters = True
evaluation = dict(interval=1, metric='bbox',classwise=True)


data = dict(
    samples_per_gpu=4
)
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001, #/8*gpu_number,
    betas=(0.9, 0.999),
    weight_decay=0.05)
 