free_index = 0
custom_imports = dict(imports=["projects.minkunet.minkunet"], allow_failed_imports=False)

# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.models.backbones.minkunet_backbone import MinkUNetBackbone
from mmdet3d.models.data_preprocessors.data_preprocessor import Det3DDataPreprocessor
from mmdet3d.models.decode_heads.minkunet_head import MinkUNetHead
from mmdet3d.models.segmentors.minkunet import MinkUNet

model = dict(
    type=MinkUNet,
    data_preprocessor=dict(
        type=Det3DDataPreprocessor,
        voxel=True,
        voxel_type="minkunet",
        batch_first=False,
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=[-100, -100, -20, 100, 100, 20],
            voxel_size=[0.2, 0.2, 0.2],
            max_voxels=(-1, -1),
        ),
    ),
    backbone=dict(
        type=MinkUNetBackbone,
        in_channels=4,
        num_stages=4,
        base_channels=32,
        encoder_channels=[32, 64, 128, 256],
        encoder_blocks=[2, 2, 2, 2],
        decoder_channels=[256, 128, 96, 96],
        decoder_blocks=[2, 2, 2, 2],
        block_type="basic",
        sparseconv_backend="torchsparse",
    ),
    decode_head=dict(
        type=MinkUNetHead,
        channels=96,
        num_classes=20,
        dropout_ratio=0,
        loss_decode=dict(type="mmdet.CrossEntropyLoss", avg_non_ignore=True),
        ignore_index=free_index,
    ),
    train_cfg=dict(),
    test_cfg=dict(),
)


from mmengine.dataset.sampler import DefaultSampler
from mmdet3d.datasets.transforms import (
    LoadPointsFromFile,
    LoadAnnotations3D,
    PointSegClassMapping,
    PointSample,
    RandomFlip3D,
    GlobalRotScaleTrans,
    LaserMix,
    PolarMix,
)
from mmdet3d.datasets.utils import Pack3DDetInputs
from mmdet3d.evaluation.metrics import SegMetric
from projects.minkunet.minkunet.semantickitti_dataset import SemanticKittiSC as dataset_type


data_root = "data/semantickitti/"

class_names = (
    "free",  # 0
    "car",  # 1
    "bicycle",  # 2
    "motorcycle",  # 3
    "truck",  # 4
    "bus",  # 5
    "person",  # 6
    "bicyclist",  # 7
    "motorcyclist",  # 8
    "road",  # 9
    "parking",  # 10
    "sidewalk",  # 11
    "other-ground",  # 12
    "building",  # 13
    "fence",  # 14
    "vegetation",  # 15
    "trunck",  # 16
    "terrian",  # 17
    "pole",  # 18
    "traffic-sign",  # 19
)
palette = list(
    [
        [0, 0, 0],
        [100, 150, 245],
        [100, 230, 245],
        [30, 60, 150],
        [80, 30, 180],
        [100, 80, 250],
        [155, 30, 30],
        [255, 40, 200],
        [150, 30, 90],
        [255, 0, 255],
        [255, 150, 255],
        [75, 0, 75],
        [175, 0, 75],
        [255, 200, 0],
        [255, 120, 50],
        [0, 175, 0],
        [135, 60, 0],
        [150, 240, 80],
        [255, 240, 150],
        [255, 0, 0],
    ]
)

labels_map = {
    0: 0,  # "unlabeled"
    1: 0,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,  # "car"
    11: 2,  # "bicycle"
    13: 5,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,  # "motorcycle"
    16: 5,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,  # "truck"
    20: 5,  # "other-vehicle"
    30: 6,  # "person"
    31: 7,  # "bicyclist"
    32: 8,  # "motorcyclist"
    40: 9,  # "road"
    44: 10,  # "parking"
    48: 11,  # "sidewalk"
    49: 12,  # "other-ground"
    50: 13,  # "building"
    51: 14,  # "fence"
    52: 0,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,  # "lane-marking" to "road" ---------------------------------mapped
    70: 15,  # "vegetation"
    71: 16,  # "trunk"
    72: 17,  # "terrain"
    80: 18,  # "pole"
    81: 19,  # "traffic-sign"
    99: 0,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,  # "moving-car" to "car" ------------------------------------mapped
    253: 7,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,  # "moving-person" to "person" ------------------------------mapped
    255: 8,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,  # "moving-truck" to "truck" --------------------------------mapped
    259: 5,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

metainfo = dict(classes=class_names, seg_label_mapping=labels_map, max_label=259)
input_modality = dict(use_lidar=True, use_camera=True)

backend_args = None

train_pipeline = [
    dict(type=LoadPointsFromFile, coord_type="LIDAR", load_dim=4, use_dim=4, backend_args=backend_args),
    dict(
        type=LoadAnnotations3D,
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype="np.int32",
        seg_offset=2**16,
        dataset_type="semantickitti",
        backend_args=backend_args,
    ),
    dict(type=PointSegClassMapping),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type=LaserMix,
                    num_areas=[3, 4, 5, 6],
                    pitch_angles=[-25, 3],
                    pre_transform=[
                        dict(type=LoadPointsFromFile, coord_type="LIDAR", load_dim=4, use_dim=4),
                        dict(
                            type=LoadAnnotations3D,
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype="np.int32",
                            seg_offset=2**16,
                            dataset_type="semantickitti",
                        ),
                        dict(type=PointSegClassMapping),
                    ],
                    prob=1,
                )
            ],
            [
                dict(
                    type=PolarMix,
                    instance_classes=[1, 2, 3, 4, 5, 6, 7, 8],
                    swap_ratio=0.5,
                    rotate_paste_ratio=1.0,
                    pre_transform=[
                        dict(type=LoadPointsFromFile, coord_type="LIDAR", load_dim=4, use_dim=4),
                        dict(
                            type=LoadAnnotations3D,
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype="np.int32",
                            seg_offset=2**16,
                            dataset_type="semantickitti",
                        ),
                        dict(type=PointSegClassMapping),
                    ],
                    prob=1,
                )
            ],
        ],
        prob=[0.5, 0.5],
    ),
    dict(type=PointSample, num_points=0.9),
    dict(
        type=RandomFlip3D,
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(
        type=GlobalRotScaleTrans,
        rot_range=[0.0, 6.28318531],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(type=Pack3DDetInputs, keys=["points", "pts_semantic_mask"]),
]
test_pipeline = [
    dict(type=LoadPointsFromFile, coord_type="LIDAR", load_dim=4, use_dim=4, backend_args=backend_args),
    dict(
        type=LoadAnnotations3D,
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype="np.int32",
        seg_offset=2**16,
        dataset_type="semantickitti",
        backend_args=backend_args,
    ),
    dict(type=PointSegClassMapping),
    dict(type=Pack3DDetInputs, keys=["points", "pts_semantic_mask"]),
]


train_split = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file="semantickittiDataset_infos_train.pkl",
    pipeline=train_pipeline,
    metainfo=metainfo,
    modality=input_modality,
    backend_args=backend_args,
    ignore_index=free_index,
)

val_split = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file="semantickittiDataset_infos_val.pkl",
    pipeline=test_pipeline,
    metainfo=metainfo,
    modality=input_modality,
    test_mode=True,
    backend_args=backend_args,
    ignore_index=free_index,
)

test_split = val_split

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=train_split,
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=val_split,
)

test_dataloader = val_dataloader


val_evaluator = dict(type=SegMetric)
test_evaluator = val_evaluator

# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import OneCycleLR
from torch.optim.adamw import AdamW

# This schedule is mainly used on Semantickitti dataset in segmentation task
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW,
        lr=0.01,
        betas=(0.9, 0.999),
        weight_decay=(0.01),
        eps=0.000005,
    ),
)

param_scheduler = [
    dict(
        type=OneCycleLR,
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
auto_scale_lr = dict(enable=True, base_batch_size=4)

randomness = dict(seed=3407)
