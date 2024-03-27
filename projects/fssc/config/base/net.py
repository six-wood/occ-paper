import torch.nn as nn
from projects.fssc.plugin.model.ssc_head import SscHead
from projects.fssc.plugin.model.sc_head import ScHead
from projects.fssc.plugin.model.auxiliary_head import PtsAuxHead
from projects.fssc.plugin.model.neck import SampleNet
from projects.fssc.plugin.model.ssc_net import SscNet
from projects.fssc.plugin.model.bev_backbone import BevNet
from projects.fssc.plugin.model.data_preprocessor import OccDataPreprocessor
from projects.fssc.plugin.model.pts_backbone import PtsNet
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet3d.models.losses import LovaszLoss
from mmdet3d.models.voxel_encoders import DynamicVFE, SegVFE, HardVFE
from mmdet3d.models.backbones import MinkUNetBackbone

from mmengine.config import read_base

with read_base():
    from .share_paramenter import *

model = dict(
    type=SscNet,
    data_preprocessor=dict(
        type=OccDataPreprocessor,
        voxel=True,
        voxel_layer=dict(
            max_num_points=64,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1),
        ),
    ),
    pts_backbone=dict(
        type=PtsNet,
        pts_voxel_encoder=dict(
            type="HardVFE",
            in_channels=4,
            feat_channels=[16],
            with_distance=False,
            voxel_size=voxel_size,
            with_cluster_center=True,
            with_voxel_center=True,
            point_cloud_range=point_cloud_range,
        ),
        in_channels=16,
        base_channels=16,
        num_stages=1,
        encoder_channels=[16],
        encoder_blocks=[1],
        block_type="basic",
    ),
    bev_backbone=dict(
        type=BevNet,
        bev_input_dimensions=32,
        bev_stem_channels=32,
        bev_num_stages=3,
        bev_stage_blocks=(4, 4, 4),
        bev_strides=(2, 2, 2),
        bev_dilations=(1, 1, 1),
        bev_encoder_out_channels=(64, 96, 128),
        bev_decoder_blocks=(1, 1, 1),
        bev_decoder_out_channels=(96, 64, 32),
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
    neck=dict(
        type=SampleNet,
        top_k_scatter=k_scatter,
        sc_embedding_dim=[16],
        voxel_size=voxel_size,
        pc_range=point_cloud_range,
    ),
    ssc_head=dict(
        type=SscHead,
        out_channels=96,
        num_classes=num_classes,
        ignore_index=ignore_index,
        voxel_net=dict(
            type=MinkUNetBackbone,
            in_channels=16,
            num_stages=4,
            base_channels=32,
            encoder_channels=[32, 64, 128, 256],
            encoder_blocks=[2, 2, 2, 2],
            decoder_channels=[256, 128, 96, 96],
            decoder_blocks=[2, 2, 2, 2],
            block_type="basic",
            sparseconv_backend="spconv",
        ),
        loss_ce=dict(
            type=CrossEntropyLoss,
            class_weight=semantickitti_class_weight,
            loss_weight=1.0,
            avg_non_ignore=True,
        ),
        loss_lovasz=dict(
            type=LovaszLoss,
            class_weight=semantickitti_class_weight,
            loss_weight=1.0,
            reduction="none",
        ),
    ),
)
