# Copyright (c) OpenMMLab. All rights reserved.

custom_imports = dict(imports=["projects.sc.plugin"], allow_failed_imports=False)
from mmengine.dataset.sampler import DefaultSampler
from mmdet3d.datasets.transforms import (
    LoadPointsFromFile,
    LoadAnnotations3D,
    PointSegClassMapping,
    PointSample,
    RandomFlip3D,
    GlobalRotScaleTrans,
)
from mmdet3d.datasets.utils import Pack3DDetInputs
from projects.sc.plugin.evaluation.ssc_metric import SscMetric
from projects.sc.plugin.datasets.semantickitti_dataset import SemanticKittiSC as dataset_type
from projects.sc.plugin.datasets.transforms_3d import SemkittiRangeView, LoadVoxelLabelFromFile

# from projects.sc.plugin.evaluation.ssc_metric import SscMetric

from mmengine.config import read_base

with read_base():
    from .share_paramenter import *

data_root = "data/semantickitti/"
# class_names = (
#     "free",
#     "occupied",
# )
# palette = list(
#     [
#         [0, 0, 0],
#         [255, 255, 255],
#     ]
# )

# labels_map = {
#     0: 0,
#     1: 1,
# }
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
        type=LoadVoxelLabelFromFile,
        grid_size=grid_size,
    ),
    dict(
        type=SemkittiRangeView,
        H=64,
        W=512,
        fov_up=3.0,
        fov_down=-25.0,
        means=(11.71279, -0.1023471, 0.4952, -1.0545, 0.2877),
        stds=(10.24, 12.295865, 9.4287, 0.8643, 0.1450),
        ignore_index=free_index,
    ),
    dict(
        type=Pack3DDetInputs,
        keys=["img", "gt_semantic_seg", "points"],
        meta_keys=("voxel_label",),
    ),
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
    dict(
        type=LoadVoxelLabelFromFile,
        grid_size=grid_size,
    ),
    dict(
        type=SemkittiRangeView,
        H=64,
        W=512,
        fov_up=3.0,
        fov_down=-25.0,
        means=(11.71279, -0.1023471, 0.4952, -1.0545, 0.2877),
        stds=(10.24, 12.295865, 9.4287, 0.8643, 0.1450),
        ignore_index=free_index,
    ),
    dict(
        type=Pack3DDetInputs,
        keys=["img", "points"],
        meta_keys=("voxel_label",),
    ),
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


val_evaluator = dict(type=SscMetric)
test_evaluator = val_evaluator
