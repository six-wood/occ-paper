import torch.nn as nn
from projects.occ_paper.mmdet3d_plugin.models.pc_bevnet import BevNet
from projects.occ_paper.mmdet3d_plugin.models.pc_rangenet import RangeNet
from projects.occ_paper.mmdet3d_plugin.models.pc_fusionnet import FusionNet
from projects.occ_paper.mmdet3d_plugin.models.ssc_net import SscNet
from projects.occ_paper.mmdet3d_plugin.models.head import DenseSscHead
from projects.occ_paper.mmdet3d_plugin.models.data_preprocessor import SccDataPreprocessor
from projects.occ_paper.mmdet3d_plugin.models.losses import OccLovaszLoss
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.backbones.resnet import ResNet
from mmdet3d.models.layers.fusion_layers import PointFusion
from mmdet3d.models.backbones import MinkUNetBackbone
from mmengine.config import read_base

with read_base():
    from .share_paramenter import *

HSwin = dict(type=nn.Hardswish, inplace=True)
ReLU = dict(type=nn.ReLU, inplace=True)
syncNorm = dict(type=nn.SyncBatchNorm, momentum=0.01, eps=1e-3)
conv2d = dict(type=nn.Conv2d)
conv3d = dict(type=nn.Conv3d)

model = dict(
    type=SscNet,
    use_pred_mask=True,
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
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32,
    ),
    # img_backbone=dict(
    #     type=ResNet,
    #     depth=50,
    #     in_channels=3,
    #     num_stages=4,
    #     out_indices=(2,),
    #     frozen_stages=1,
    #     norm_cfg=syncNorm,
    #     norm_eval=True,
    #     style="pytorch",
    #     init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    # ),
    pts_bev_backbone=dict(
        type=BevNet,
        conv_cfg=conv2d,
        act_cfg=HSwin,
        norm_cfg=syncNorm,
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
            fuse_channel,
            range_out_channel,
        ),
        conv_cfg=conv2d,
        act_cfg=HSwin,
        norm_cfg=syncNorm,
    ),
    fusion_neck=dict(
        type=FusionNet,
        conv_cfg=conv3d,
        norm_cfg=syncNorm,
        act_cfg=HSwin,
    ),
    ssc_head=dict(
        type=DenseSscHead,
        seg_channels=128,
        num_classes=num_classes,
        sem_sparse_backbone=dict(
            type=MinkUNetBackbone,
            in_channels=128,
            num_stages=2,
            base_channels=128,
            encoder_channels=[128, 256],
            encoder_blocks=[2, 2],
            decoder_channels=[256, 128],
            decoder_blocks=[2, 2],
            block_type="basic",
            sparseconv_backend="spconv",
        ),
        loss_geo=dict(
            type=CrossEntropyLoss,
            class_weight=geo_class_weight,
            loss_weight=1.0,
        ),
        loss_sem=dict(
            type=CrossEntropyLoss,
            class_weight=semantickitti_class_weight,
            loss_weight=1.0,
        ),
        loss_lovasz=dict(
            type=OccLovaszLoss,
            classes=class_index,  # ignore the free class
            class_weight=semantickitti_class_weight,
            reduction="none",
            loss_weight=1.0,
        ),
        ignore_index=ignore_index,
        free_index=free_index,
        conv_cfg=conv3d,
        norm_cfg=syncNorm,
        act_cfg=HSwin,
    ),
)
