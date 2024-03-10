from projects.CENet.cenet.boundary_loss import BoundaryLoss
from projects.CENet.cenet.cenet_backbone import CENet
from projects.CENet.cenet.range_image_head import RangeImageHead
from projects.CENet.cenet.range_image_segmentor import RangeImageSegmentor
from projects.CENet.cenet.transforms_3d import SemkittiRangeView

with read_base():
    from mmdet3d.configs._base_.default_runtime import *
    from .base.lr import *
    from .base.semankitti import *

custom_imports = dict(imports=["projects.occ_paper.mmdet3d_plugin"], allow_failed_imports=False)

# For SemanticKitti we usually do 19-class segmentation.
# For labels_map we follow the uniform format of MMDetection & MMSegmentation
# i.e. we consider the unlabeled class as the last one, which is different
# from the original implementation of some methods e.g. Cylinder3D.

backend_args = None

train_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4, backend_args=backend_args),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype="np.int32",
        seg_offset=2**16,
        dataset_type="semantickitti",
        backend_args=backend_args,
    ),
    dict(type="PointSegClassMapping"),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="LaserMix",
                    num_areas=[3, 4, 5, 6],
                    pitch_angles=[-25, 3],
                    pre_transform=[
                        dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
                        dict(
                            type="LoadAnnotations3D",
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype="np.int32",
                            seg_offset=2**16,
                            dataset_type="semantickitti",
                        ),
                        dict(type="PointSegClassMapping"),
                    ],
                    prob=1,
                )
            ],
            [
                dict(
                    type="PolarMix",
                    instance_classes=[1, 2, 3, 4, 5, 6, 7, 8],
                    swap_ratio=0.5,
                    rotate_paste_ratio=1.0,
                    pre_transform=[
                        dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
                        dict(
                            type="LoadAnnotations3D",
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype="np.int32",
                            seg_offset=2**16,
                            dataset_type="semantickitti",
                        ),
                        dict(type="PointSegClassMapping"),
                    ],
                    prob=1,
                )
            ],
        ],
        prob=[0.5, 0.5],
    ),
    dict(type="PointSample", num_points=0.9),
    dict(type="RandomFlip3D", sync_2d=False, flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    dict(
        type="GlobalRotScaleTrans",
        rot_range=[-3.1415929, 3.1415929],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(
        type=SemkittiRangeView,
        H=64,
        W=512,
        fov_up=3.0,
        fov_down=-25.0,
        means=(11.71279, -0.1023471, 0.4952, -1.0545, 0.2877),
        stds=(10.24, 12.295865, 9.4287, 0.8643, 0.1450),
        ignore_index=0,
    ),
    dict(type="Pack3DDetInputs", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4, backend_args=backend_args),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype="np.int32",
        seg_offset=2**16,
        dataset_type="semantickitti",
        backend_args=backend_args,
    ),
    dict(type="PointSegClassMapping"),
    dict(
        type=SemkittiRangeView,
        H=64,
        W=512,
        fov_up=3.0,
        fov_down=-25.0,
        means=(11.71279, -0.1023471, 0.4952, -1.0545, 0.2877),
        stds=(10.24, 12.295865, 9.4287, 0.8643, 0.1450),
        ignore_index=0,
    ),
    dict(type="Pack3DDetInputs", keys=["img"], meta_keys=("proj_x", "proj_y", "proj_range", "unproj_range")),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="semantickittiDataset_infos_train.pkl",
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=0,
        backend_args=backend_args,
    ),
)
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="semantickittiDataset_infos_val.pkl",
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=0,
        test_mode=True,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="SegMetric")
test_evaluator = val_evaluator

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="Det3DLocalVisualizer", vis_backends=vis_backends, name="visualizer")

model = dict(
    type=RangeImageSegmentor,
    data_preprocessor=dict(type="Det3DDataPreprocessor"),
    backbone=dict(
        type=CENet,
        in_channels=5,
        stem_channels=128,
        num_stages=4,
        stage_blocks=(3, 4, 6, 3),
        out_channels=(128, 128, 128, 128),
        fuse_channels=(256, 128),
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        act_cfg=dict(type="HSwish", inplace=True),
    ),
    decode_head=dict(
        type=RangeImageHead,
        channels=128,
        num_classes=20,
        dropout_ratio=0,
        loss_ce=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=False, class_weight=None, loss_weight=1.0),
        loss_lovasz=dict(type="LovaszLoss", loss_weight=1.5, reduction="none"),
        loss_boundary=dict(type="BoundaryLoss", loss_weight=1.0),
        conv_seg_kernel_size=1,
        ignore_index=0,
    ),
    auxiliary_head=[
        dict(
            type=RangeImageHead,
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=False, class_weight=None, loss_weight=1.0),
            loss_lovasz=dict(type="LovaszLoss", loss_weight=1.5, reduction="none"),
            loss_boundary=dict(type="BoundaryLoss", loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=0,
            indices=2,
        ),
        dict(
            type=RangeImageHead,
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=False, class_weight=None, loss_weight=1.0),
            loss_lovasz=dict(type="LovaszLoss", loss_weight=1.5, reduction="none"),
            loss_boundary=dict(type="BoundaryLoss", loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=0,
            indices=3,
        ),
        dict(
            type=RangeImageHead,
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=False, class_weight=None, loss_weight=1.0),
            loss_lovasz=dict(type="LovaszLoss", loss_weight=1.5, reduction="none"),
            loss_boundary=dict(type="BoundaryLoss", loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=0,
            indices=4,
        ),
    ],
    train_cfg=None,
    test_cfg=dict(use_knn=True, knn=7, search=7, sigma=1.0, cutoff=2.0),
)

# optimizer
# This schedule is mainly used on Semantickitti dataset in segmentation task
optim_wrapper = dict(
    type="AmpOptimWrapper", loss_scale="dynamic", optimizer=dict(type="AdamW", lr=0.04, betas=(0.9, 0.999), weight_decay=(0.01), eps=0.000005)
)

param_scheduler = [
    dict(
        type="OneCycleLR",
        total_steps=50,
        by_epoch=True,
        eta_max=0.0025,
        pct_start=0.2,
        div_factor=25.0,
        final_div_factor=100.0,
        convert_to_iter_based=True,
    )
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (4 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)

default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=1))
