import torch.nn as nn
from projects.fssc.plugin.model.ssc_head import SscHead
from projects.fssc.plugin.model.sc_head import ScHead
from projects.fssc.plugin.model.neck import DownsampleNet
from projects.fssc.plugin.model.ssc_net import SscNet
from projects.fssc.plugin.model.bev_backbone import BevNet
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.models.losses import LovaszLoss

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
        type=DownsampleNet,
        top_k_scatter=8,
        voxel_size=voxel_size,
        pc_range=point_cloud_range,
    ),
    ssc_head=dict(
        type=SscHead,
        loss_ce=dict(
            type=CrossEntropyLoss,
            class_weight=semantickitti_class_weight,
            ignore_index=ignore_index,
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
