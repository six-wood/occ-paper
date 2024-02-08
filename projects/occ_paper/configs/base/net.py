import torch.nn as nn
from projects.occ_paper.mmdet3d_plugin.models.pc_bevnet import BevNet
from projects.occ_paper.mmdet3d_plugin.models.pc_rangenet import RangeNet
from projects.occ_paper.mmdet3d_plugin.models.pc_fusionnet import FusionNet
from projects.occ_paper.mmdet3d_plugin.models.ssc_net import SscNet
from projects.occ_paper.mmdet3d_plugin.models.head import DenseSscHead
from projects.occ_paper.mmdet3d_plugin.models.loss import OccLovaszLoss, Geo_scal_loss, Sem_scal_loss
from projects.occ_paper.mmdet3d_plugin.models.data_preprocessor import SccDataPreprocessor
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.losses.focal_loss import FocalLoss
from mmengine.config import read_base

HSwin = dict(type=nn.Hardswish, inplace=True)
ReLU = dict(type=nn.ReLU, inplace=True)

with read_base():
    from .share_paramenter import *

model = dict(
    type=SscNet,
    data_preprocessor=dict(
        type=SccDataPreprocessor,
        voxel=True,
        voxel_type="dynamic",
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1),
        ),
        range_img=True,
        range_layer=dict(
            H=64,
            W=1024,
            fov_up=3.0,
            fov_down=-25.0,
            fov_left=-90.0,
            fov_right=90.0,
            means=(11.71279, -0.1023471, 0.4952, -1.0545, 0.2877),
            stds=(10.24, 12.295865, 9.4287, 0.8643, 0.1450),
        ),
    ),
    pts_bev_backbone=dict(
        type=BevNet,
        act_cfg=HSwin,
    ),
    pts_range_backbone=dict(
        type=RangeNet,
        in_channels=5,
        stem_channels=range_encoder_channel,
        num_stages=4,
        stage_blocks=(3, 4, 6, 3),
        out_channels=(
            range_encoder_channel,
            range_encoder_channel,
            range_encoder_channel,
            range_encoder_channel,
        ),
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        fuse_channels=(
            2 * range_encoder_channel,
            fuse_channel,
        ),
        act_cfg=HSwin,
    ),
    pts_fusion_neck=dict(
        type=FusionNet,
        conv_cfg=dict(type=nn.Conv2d),
        norm_cfg=dict(type=nn.BatchNorm2d),
    ),
    pts_ssc_head=dict(
        type=DenseSscHead,
        inplanes=fuse_channel,
        planes=fuse_channel,
        nbr_classes=number_classes,
        dilations_conv_list=[1, 2, 3],
        # loss_focal=dict(
        #     type=FocalLoss,
        #     use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=1.0,
        # ),
        loss_ce=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            loss_weight=1.0,
        ),
        loss_lovasz=dict(
            type=OccLovaszLoss,
            loss_weight=1.0,
            reduction="none",
        ),
        loss_geo=dict(
            type=Geo_scal_loss,
            ignore_index=ignore_index,
            free_index=free_index,
            loss_weight=1.0,
        ),
        loss_sem=dict(
            type=Sem_scal_loss,
            ignore_index=ignore_index,
            loss_weight=1.0,
        ),
        class_frequencies=semantic_kitti_class_frequencies,
        ignore_index=ignore_index,
        conv_cfg=dict(type=nn.Conv3d),
        norm_cfg=dict(type=nn.BatchNorm3d),
        act_cfg=HSwin,
    ),
)
