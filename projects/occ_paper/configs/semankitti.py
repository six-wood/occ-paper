# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms.processing import TestTimeAug
from mmengine.dataset.sampler import DefaultSampler

from mmdet3d.datasets.transforms.loading import (
    LoadImageFromFile,
    LoadPointsFromFile,
    LoadAnnotations3D,
)
from mmdet3d.models.segmentors.seg3d_tta import Seg3DTTAModel

from projects.occ_paper.occ_paper.loading import LoadVoxelLabelFromFile
from projects.occ_paper.occ_paper.transforms_3d import ApplayVisMask, SemkittiRangeView
from projects.occ_paper.occ_paper.formating import PackOccInputs
from projects.occ_paper.occ_paper.semantickitti_dataset import (
    SemanticKittiSC as dataset_type,
)

from projects.occ_paper.occ_paper.ssc_metric import SSCMetric
from mmengine.config import read_base

with read_base():
    from .share_paramenter import *

data_root = "data/semantickitti/"
class_names = (
    "free",
    "occupied",
)
palette = list(
    [
        [0, 0, 0],
        [255, 255, 255],
    ]
)

labels_map = {
    0: 0,
    1: 1,
}

metainfo = dict(classes=class_names, seg_label_mapping=labels_map, max_label=259)
input_modality = dict(use_lidar=True, use_camera=True)

backend_args = None

train_pipeline = [
    dict(type=LoadPointsFromFile, coord_type="LIDAR", load_dim=4, use_dim=4, backend_args=backend_args),
    dict(type=LoadVoxelLabelFromFile, grid_size=[256, 256, 32], scale=scale),
    dict(type=SemkittiRangeView),
    dict(type=ApplayVisMask),
    dict(
        type=PackOccInputs,
        keys=["points", "voxel_label"],
    ),
]

test_pipeline = [
    dict(type=LoadPointsFromFile, coord_type="LIDAR", load_dim=4, use_dim=4, backend_args=backend_args),
    dict(type=LoadVoxelLabelFromFile, grid_size=[256, 256, 32], scale=scale),
    dict(type=SemkittiRangeView),
    dict(type=ApplayVisMask),
    dict(
        type=PackOccInputs,
        keys=["points", "voxel_label"],
    ),
]

val_pipeline = train_pipeline

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
    ann_file="semantickittiDataset_infos_test.pkl.pkl",
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

val_evaluator = dict(type=SSCMetric)
test_evaluator = val_evaluator
