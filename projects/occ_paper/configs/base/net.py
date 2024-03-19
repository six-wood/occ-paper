import torch.nn as nn
from projects.occ_paper.mmdet3d_plugin.models.bevnet import BevNet
from projects.occ_paper.mmdet3d_plugin.models.rangenet import RangeNet
from projects.occ_paper.mmdet3d_plugin.models.fusionet import FusionNet
from projects.occ_paper.mmdet3d_plugin.models.ssc_net import SscNet
from projects.occ_paper.mmdet3d_plugin.models.head import DenseSscHead
from projects.occ_paper.mmdet3d_plugin.models.data_preprocessor import SccDataPreprocessor, Det3DDataPreprocessor
from projects.occ_paper.mmdet3d_plugin.models.losses import OccLovaszLoss, BoundLoss
from projects.occ_paper.mmdet3d_plugin.models.rangehead import RangeHead
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.backbones.resnet import ResNet
from mmdet3d.models.layers.fusion_layers import PointFusion
from mmdet3d.models.backbones import MinkUNetBackbone
from mmdet3d.models.losses import LovaszLoss
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
    bev_backbone=dict(
        type=BevNet,
        conv_cfg=conv2d,
        act_cfg=HSwin,
        norm_cfg=syncNorm,
    ),
    fusion_neck=dict(
        type=FusionNet,
        top_k_scatter=k_scatter,
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
            in_channels=3,
            num_stages=4,
            base_channels=32,
            encoder_channels=[32, 64, 128, 256],
            encoder_blocks=[2, 2, 2, 2],
            decoder_channels=[96, 96, 256, 128],
            decoder_blocks=[2, 2, 2, 2],
            block_type="basic",
            sparseconv_backend="spconv",
        ),
        # check the class weight
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
        # loss_lovasz=dict(
        #     type=OccLovaszLoss,
        #     classes=class_index,
        #     class_weight=semantickitti_class_weight,
        #     reduction="none",
        #     loss_weight=1.5,
        # ),
        ignore_index=ignore_index,
        free_index=free_index,
        conv_cfg=conv3d,
        norm_cfg=syncNorm,
        act_cfg=HSwin,
    ),
    auxiliary_head=[
        # dict(
        #     type=RangeHead,
        #     channels=128,
        #     num_classes=num_classes,
        #     dropout_ratio=0,
        #     loss_ce=dict(
        #         type=CrossEntropyLoss,
        #         class_weight=semantickitti_class_weight,
        #         loss_weight=1.0,
        #     ),
        #     loss_lovasz=dict(
        #         type=LovaszLoss,
        #         classes=class_index,
        #         class_weight=semantickitti_class_weight,
        #         loss_weight=1.5,
        #         reduction="none",
        #     ),
        #     loss_boundary=dict(type=BoundLoss, loss_weight=1.0),
        #     conv_seg_kernel_size=1,
        #     ignore_index=0,
        #     indices=0,
        # ),
        # dict(
        #     type=RangeHead,
        #     channels=128,
        #     num_classes=num_classes,
        #     dropout_ratio=0,
        #     loss_ce=dict(type=CrossEntropyLoss, use_sigmoid=False, class_weight=None, loss_weight=1.0),
        #     loss_lovasz=dict(type=LovaszLoss, loss_weight=1.5, reduction="none"),
        #     loss_boundary=dict(type=BoundLoss, loss_weight=1.0),
        #     conv_seg_kernel_size=1,
        #     ignore_index=0,
        #     indices=1,
        # ),
        # dict(
        #     type=RangeHead,
        #     channels=128,
        #     num_classes=num_classes,
        #     dropout_ratio=0,
        #     loss_ce=dict(type=CrossEntropyLoss, use_sigmoid=False, class_weight=None, loss_weight=1.0),
        #     loss_lovasz=dict(type=LovaszLoss, loss_weight=1.5, reduction="none"),
        #     loss_boundary=dict(type=BoundLoss, loss_weight=1.0),
        #     conv_seg_kernel_size=1,
        #     ignore_index=0,
        #     indices=2,
        # ),
        # dict(
        #     type=RangeHead,
        #     channels=128,
        #     num_classes=num_classes,
        #     dropout_ratio=0,
        #     loss_ce=dict(type=CrossEntropyLoss, use_sigmoid=False, class_weight=None, loss_weight=1.0),
        #     loss_lovasz=dict(type=LovaszLoss, loss_weight=1.5, reduction="none"),
        #     loss_boundary=dict(type=BoundLoss, loss_weight=1.0),
        #     conv_seg_kernel_size=1,
        #     ignore_index=0,
        #     indices=3,
        # ),
        # dict(
        #     type=RangeHead,
        #     channels=128,
        #     num_classes=num_classes,
        #     dropout_ratio=0,
        #     loss_ce=dict(type=CrossEntropyLoss, use_sigmoid=False, class_weight=None, loss_weight=1.0),
        #     loss_lovasz=dict(type=LovaszLoss, loss_weight=1.5, reduction="none"),
        #     loss_boundary=dict(type=BoundLoss, loss_weight=1.0),
        #     conv_seg_kernel_size=1,
        #     ignore_index=0,
        #     indices=4,
        # ),
    ],
)
