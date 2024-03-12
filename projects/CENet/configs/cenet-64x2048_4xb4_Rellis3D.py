from mmengine.config import read_base

# 纯python表达方式
with read_base():
    from .cenet_64x512_4xb4_Rellis3d import *

backend_args = None
train_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type=LoadAnnotations3D,
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type=Rellis3D,
        backend_args=backend_args),
    dict(type=PointSegClassMapping),
    dict(type=PointSample, num_points=0.9),
    dict(
        type=RandomFlip3D,
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type=GlobalRotScaleTrans,
        rot_range=[-3.1415929, 3.1415929],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(
        type=SemkittiRangeView,
        H=64,
        W=2048,
        fov_up=22.5,
        fov_down=-22.5,
        means=(4.84649722, -0.187910314, 0.193718327, -0.246564824, 0.00260723157),
        stds=(6.05381850, 5.61048984, 5.27298844, 0.849105890, 0.00284712457),
        ignore_index=0),
    dict(type=Pack3DDetInputs, keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type=LoadAnnotations3D,
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type=Rellis3D,
        backend_args=backend_args),
    dict(type=PointSegClassMapping),
    dict(
        type=SemkittiRangeView,
        H=64,
        W=2048,
        fov_up=22.5,
        fov_down=-22.5,
        means=(4.84649722, -0.187910314, 0.193718327, -0.246564824, 0.00260723157),
        stds=(6.05381850, 5.61048984, 5.27298844, 0.849105890, 0.00284712457),
        ignore_index=0),
    dict(
        type=Pack3DDetInputs,
        keys=['img'],
        meta_keys=('proj_x', 'proj_y', 'proj_range', 'unproj_range'))
]

train_dataloader.update(dataset=dict(pipeline=train_pipeline))
val_dataloader.update(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

default_hooks.update(checkpoint=dict(type=CheckpointHook, interval=1))

vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]
visualizer.update(
    type=Det3DLocalVisualizer, vis_backends=vis_backends, name='visualizer')
