# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms.processing import TestTimeAug
from mmengine.dataset.sampler import DefaultSampler

from mmdet3d.datasets.transforms.loading import (
    LoadImageFromFile,
    LoadPointsFromFile,
    LoadAnnotations3D,
)
from mmdet3d.models.segmentors.seg3d_tta import Seg3DTTAModel
from projects.occ_paper.mmdet3d_plugin.datasets.transforms.formating import PackSscInputs
from projects.occ_paper.mmdet3d_plugin.datasets.transforms.loading import LoadVoxelLabelFromFile
from projects.occ_paper.mmdet3d_plugin.datasets.transforms.transforms_3d import ApplayVisMask, RandomFlipVoxel
from projects.occ_paper.mmdet3d_plugin.datasets.semantickitti_dataset import SemanticKittiSC as dataset_type

from projects.occ_paper.mmdet3d_plugin.evaluation.ssc_metric import SSCMetric
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
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    18: 18,
    19: 19,
}

metainfo = dict(classes=class_names, seg_label_mapping=labels_map, max_label=259)
input_modality = dict(use_lidar=True, use_camera=True)

backend_args = None

train_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(
        type=LoadVoxelLabelFromFile,
        task="ssc",
        scale=scale,
        ignore_index=ignore_index,
        grid_size=grid_size,
    ),
    dict(
        type=ApplayVisMask,
        center=[0, 0, 0],
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        fov=fov_vertical,
    ),
    # dict(
    #     type=RandomFlipVoxel,
    #     flip_ratio_bev_horizontal=0.5,
    # ),
    dict(
        type=PackSscInputs,
        keys=["points", "voxel_label"],
    ),
]

val_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(
        type=LoadVoxelLabelFromFile,
        task="ssc",
        scale=scale,
        ignore_index=ignore_index,
        grid_size=grid_size,
    ),
    dict(
        type=PackSscInputs,
        keys=["points", "voxel_label"],
    ),
]

test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(
        type=PackSscInputs,
        keys=["points"],
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
    ignore_index=ignore_index,
)

val_split = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file="semantickittiDataset_infos_val.pkl",
    pipeline=val_pipeline,
    metainfo=metainfo,
    modality=input_modality,
    test_mode=True,
    backend_args=backend_args,
    ignore_index=ignore_index,
)

test_split = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file="semantickittiDataset_infos_test.pkl",
    pipeline=test_pipeline,
    metainfo=metainfo,
    modality=input_modality,
    test_mode=True,
    backend_args=backend_args,
    ignore_index=ignore_index,
)

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=train_split,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=val_split,
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=test_split,
)

val_evaluator = dict(type=SSCMetric, ignore_index=[ignore_index, free_index])
test_evaluator = val_evaluator
