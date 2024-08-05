_base_ = [
    './_base_/models/segformer_mit-b0.py', './_base_/datasets/hsi_facade.py',
    './_base_/default_runtime.py', './_base_/schedules/schedule_epoch.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(num_classes=44))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=True, begin=0, end=10),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=10,
        end=200,
        by_epoch=True,
    )
]
train_dataloader = dict(batch_size=32, num_workers=4)
val_dataloader = dict(batch_size=16, num_workers=4)
test_dataloader = val_dataloader
