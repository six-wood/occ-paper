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
from mmdet3d.models.voxel_encoders import DynamicVFE

from mmengine.config import read_base

with read_base():
    from .share_paramenter import *

model = dict(
    type=SscNet,
    data_preprocessor=dict(
        type=OccDataPreprocessor,
        voxel=True,
        voxel_type="dynamic",
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1),
        ),
    ),
    pts_backbone=dict(
        type=PtsNet,
        pts_voxel_encoder=dict(
            type=DynamicVFE,
            in_channels=4,
            feat_channels=[16],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            with_distance=False,
            with_cluster_center=True,
            with_voxel_center=True,
        ),
        in_channels=16,
        base_channels=32,
        num_stages=1,
        encoder_channels=[32],
        encoder_blocks=[1],
        decoder_channels=[32],
        decoder_blocks=[1],
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
        sc_embedding_dim=[16, 32],
        voxel_size=voxel_size,
        pc_range=point_cloud_range,
    ),
    ssc_head=dict(
        type=SscHead,
        in_channels=32,
        ignore_index=ignore_index,
        mid_channels=32,
        num_classes=num_classes,
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
    # aux_head=dict(
    #     type=PtsAuxHead,
    #     channels=sc_embedding_dim,
    #     num_classes=num_classes,
    #     ignore_index=free_index,
    #     dropout_ratio=0.0,
    #     loss_decode=dict(
    #         type=CrossEntropyLoss,
    #         class_weight=semantickitti_class_weight,
    #         loss_weight=1.0,
    #         avg_non_ignore=True,
    #     ),
    # ),
)
