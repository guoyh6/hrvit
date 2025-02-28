# model settings
checkpoint = None
out_indices = [3, 7, 11]  # [7, 15, 23]
num_layers = 12
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
img_size = 96
patch_size = 8
in_channels = 1024
thresh = 1.0
num_classes = 14
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (96, 96, 96)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    # mean=[123.675, 116.28, 103.53],
    # std=[58.395, 57.12, 57.375],
    mean=[127.5, 127.5, 127.5], 
    std=[127.5, 127.5, 127.5],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,)
model = dict(
    type='EncoderDecoderPrune',
    pretrained=None,
    backbone=dict(
        type='ViT_prune',
        num_classes=num_classes,
        img_size=crop_size,
        patch_size=patch_size,
        in_channels=1,
        embed_dims=in_channels,
        num_layers=num_layers,
        num_heads=16,
        drop_path_rate=0.3,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        out_indices=out_indices,
        final_norm=False,
        norm_cfg=backbone_norm_cfg,
        with_cls_token=False,
        interpolate_mode='bicubic',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint,
            prefix=None,
        )
    ),
    decode_head=dict(
        type='PruneHead',
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        channels=in_channels//2,
        num_classes=num_classes,
        thresh=thresh,
        num_heads=8,
        layers_per_decoder=3,
        loss_decode=dict(
            type='ATMLoss', num_classes=1, dec_layers=1, loss_weight=1.0),
    ),
    auxiliary_head=[
        dict(
            type='PruneHead',
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            channels=in_channels//2,
            num_classes=num_classes,
            thresh=thresh,
            num_heads=8,
            layers_per_decoder=3,
            in_index=0,
            loss_decode=dict(
                type='ATMLoss', num_classes=1, dec_layers=1),
        ),
        dict(
            type='PruneHead',
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            channels=in_channels//2,
            num_classes=num_classes,
            thresh=thresh,
            num_heads=8,
            layers_per_decoder=3,
            in_index=1,
            loss_decode=dict(
                type='ATMLoss', num_classes=1, dec_layers=1),
        ),
    ],
    # test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(320, 320)),
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
