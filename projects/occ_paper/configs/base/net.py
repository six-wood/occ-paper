import torch.nn as nn
from projects.occ_paper.mmdet3d_plugin.models.pc_bevnet import BevNet
from projects.occ_paper.mmdet3d_plugin.models.pc_rangenet import RangeNet
from projects.occ_paper.mmdet3d_plugin.models.pc_fusionnet import FusionNet
from projects.occ_paper.mmdet3d_plugin.models.ssc_net import SscNet
from projects.occ_paper.mmdet3d_plugin.models.head import DenseSscHead
from projects.occ_paper.mmdet3d_plugin.models.data_preprocessor import SccDataPreprocessor
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmengine.config import read_base

with read_base():
    from .share_paramenter import *


HSwin = dict(type=nn.Hardswish, inplace=True)
ReLU = dict(type=nn.ReLU, inplace=True)
class_weight = [0.45, 0.55]
range_encoder_channel = 32
fuse_channel = 8

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
        num_stages=3,
        stage_blocks=(2, 2, 2),
        out_channels=(
            range_encoder_channel,
            range_encoder_channel,
            range_encoder_channel,
        ),
        strides=(2, 2, 2),
        dilations=(1, 1, 1),
        fuse_channels=(
            range_encoder_channel,
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
        loss_decode=dict(
            type=CrossEntropyLoss,
            avg_non_ignore=True,
            class_weight=class_weight,
        ),
        ignore_index=255,
        conv_cfg=dict(type=nn.Conv3d),
        norm_cfg=dict(type=nn.BatchNorm3d),
        act_cfg=HSwin,
    ),
)
