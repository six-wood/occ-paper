import torch.nn as nn
from projects.Occ.plugin.model.boundary_loss import BoundaryLoss
from projects.Occ.plugin.model.range_image_head import RangeImageHead
from projects.Occ.plugin.model.sc_head import ScHead
from projects.Occ.plugin.model.range_image_segmentor import SscNet
from projects.Occ.plugin.model.cenet_backbone import CENet
from projects.Occ.plugin.model.bev_backbone import BevNet
from projects.Occ.plugin.model.neck import FusionNet
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.models.losses import LovaszLoss
from projects.Occ.plugin.model.sparse_backbone import SparseBackbone
from projects.Occ.plugin.model.ssc_head import SscHead

from mmengine.config import read_base

with read_base():
    from .share_paramenter import *

model = dict(
    type=SscNet,
    data_preprocessor=dict(
        type=Det3DDataPreprocessor,
        voxel=True,
        voxel_type="dynamic",
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1),
        ),
    ),
    bev_backbone=dict(
        type=BevNet,
        bev_input_dimensions=32,
        bev_stem_channels=32,
        bev_num_stages=3,
        bev_stage_blocks=(1, 1, 1),
        bev_strides=(2, 2, 2),
        bev_dilations=(1, 1, 1),
        bev_encoder_out_channels=(48, 64, 80),
        bev_decoder_out_channels=(64, 48, 32),
        act_cfg=dict(type=nn.Hardswish, inplace=True),
    ),
    neck=dict(
        type=FusionNet,
        indices=0,
        pose_embending_dim=128,
        voxel_size=voxel_size,
        pc_range=point_cloud_range,
    ),
    sparse_backbone=dict(
        type=SparseBackbone,
        in_channels=3,
        num_stages=4,
        base_channels=32,
        encoder_channels=[32, 64, 128, 256],
        encoder_blocks=[2, 2, 2, 2],
        decoder_channels=[256, 128, 96, 96],
        decoder_blocks=[2, 2, 2, 2],
        block_type="basic",
        sparseconv_backend="spconv",
    ),
    sc_head=dict(
        type=ScHead,
        loss_ce=dict(
            type=CrossEntropyLoss,
            class_weight=geo_class_weight,
            ignore_index=ignore_index,
            loss_weight=1.0,
            avg_non_ignore=True,
        ),
    ),
    ssc_head=dict(
        type=SscHead,
        channels=96,
        num_classes=20,
        dropout_ratio=0,
        ignore_index=ignore_index,
        loss_ce=dict(
            type=CrossEntropyLoss,
            class_weight=semantickitti_class_weight,
            loss_weight=1.0,
            avg_non_ignore=True,
        ),
        loss_lovasz=dict(
            type=LovaszLoss,
            classes=class_index,  # ignore the first class
            class_weight=semantickitti_class_weight,
            reduction="none",
            loss_weight=1.0,
        ),
    ),
)
