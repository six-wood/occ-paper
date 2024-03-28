import torch.nn as nn
from projects.fssc.plugin.model.ssc_head import SscHead
from projects.fssc.plugin.model.sc_head import ScHead
from projects.fssc.plugin.model.neck import DualNet
from projects.fssc.plugin.model.ssc_net import SscNet
from projects.fssc.plugin.model.bev_backbone import BevNet
from projects.fssc.plugin.model.lovasz_loss import OccLovaszLoss
from projects.fssc.plugin.model.data_preprocessor import OccDDataPreprocessor
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.losses.focal_loss import FocalLoss
from mmdet3d.models.voxel_encoders import DynamicVFE

from mmengine.config import read_base

with read_base():
    from .share_paramenter import *

model = dict(
    type=SscNet,
    data_preprocessor=dict(
        type=OccDDataPreprocessor,
        voxel=True,
        voxel_type="dynamic",
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1),
        ),
    ),
    pts_voxel_encoder=dict(
        type=DynamicVFE,
        in_channels=4,
        feat_channels=[base_embeding_dim],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
    ),
    bev_backbone=dict(
        type=BevNet,
        bev_input_dimensions=32,
        bev_stem_channels=32,
        bev_num_stages=3,
        bev_stage_blocks=(4, 4, 4),
        bev_strides=(2, 2, 2),
        bev_dilations=(1, 1, 1),
        bev_encoder_out_channels=(48, 64, 80),
        bev_decoder_out_channels=(64, 48, 32),
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
        type=DualNet,
        top_k_scatter=k_scatter,
        sc_embedding_dim=base_embeding_dim,
        voxel_size=voxel_size,
        pc_range=point_cloud_range,
    ),
    ssc_head=dict(
        type=SscHead,
        in_channels=base_embeding_dim,
        mid_channels=32,
        num_classes=num_classes,
        loss_ce=dict(
            type=CrossEntropyLoss,
            class_weight=semantickitti_class_weight,
            ignore_index=ignore_index,
            loss_weight=1.0,
            avg_non_ignore=True,
        ),
        loss_lovasz=dict(
            type=OccLovaszLoss,
            class_weight=semantickitti_class_weight,
            loss_weight=1.0,
            reduction="none",
        ),
    ),
)
