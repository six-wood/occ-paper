import torch.nn as nn
from projects.Occ.plugin.model.boundary_loss import BoundaryLoss
from projects.Occ.plugin.model.range_image_head import RangeImageHead
from projects.Occ.plugin.model.range_image_segmentor import RangeImageSegmentor
from projects.Occ.plugin.model.cenet_backbone import CENet
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.models.losses import LovaszLoss

from mmengine.config import read_base

with read_base():
    from .share_paramenter import *

model = dict(
    type=RangeImageSegmentor,
    data_preprocessor=dict(
        type=Det3DDataPreprocessor,
        voxel_type="dynamic",
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1),
        ),
    ),
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
        act_cfg=dict(type=nn.Hardswish, inplace=True),
    ),
    decode_head=dict(
        type=RangeImageHead,
        channels=128,
        num_classes=20,
        dropout_ratio=0,
        loss_ce=dict(type=CrossEntropyLoss, use_sigmoid=False, class_weight=None, loss_weight=1.0),
        loss_lovasz=dict(type=LovaszLoss, loss_weight=1.5, reduction="none"),
        loss_boundary=dict(type=BoundaryLoss, loss_weight=1.0),
        conv_seg_kernel_size=1,
        ignore_index=free_index,
    ),
    auxiliary_head=[
        dict(
            type=RangeImageHead,
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(type=CrossEntropyLoss, use_sigmoid=False, class_weight=None, loss_weight=1.0),
            loss_lovasz=dict(type=LovaszLoss, loss_weight=1.5, reduction="none"),
            loss_boundary=dict(type=BoundaryLoss, loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=free_index,
            indices=2,
        ),
        dict(
            type=RangeImageHead,
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(type=CrossEntropyLoss, use_sigmoid=False, class_weight=None, loss_weight=1.0),
            loss_lovasz=dict(type=LovaszLoss, loss_weight=1.5, reduction="none"),
            loss_boundary=dict(type=BoundaryLoss, loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=free_index,
            indices=3,
        ),
        dict(
            type=RangeImageHead,
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(type=CrossEntropyLoss, use_sigmoid=False, class_weight=None, loss_weight=1.0),
            loss_lovasz=dict(type=LovaszLoss, loss_weight=1.5, reduction="none"),
            loss_boundary=dict(type=BoundaryLoss, loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=free_index,
            indices=4,
        ),
    ],
    train_cfg=None,
    test_cfg=dict(use_knn=True, knn=7, search=7, sigma=1.0, cutoff=2.0),
)
