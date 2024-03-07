import torch.nn as nn
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from typing import List, Dict

import torch
import numpy as np
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.utils.typing_utils import ConfigType
from mmdet3d.structures.det3d_data_sample import SampleList

from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmdet.models.dense_heads import MaskFormerHead
from .utils import ASPP3D


@MODELS.register_module()
class DenseSscHead(BaseModule):
    """
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    """

    # TODO ADD SCPNet completion Head

    def __init__(
        self,
        num_classes: int,
        seg_channels: int,
        sem_sparse_backbone: ConfigType = None,
        # loss_focal: ConfigType = None,
        loss_geo: ConfigType = None,
        loss_sem: ConfigType = None,
        loss_lovasz: ConfigType = None,
        conv_cfg: ConfigType = dict(type="Conv3d"),
        norm_cfg: ConfigType = dict(type="BN3d"),
        act_cfg: ConfigType = dict(type="ReLU"),
        ignore_index: int = 255,
        free_index: int = 0,
    ):
        super(DenseSscHead, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.ignore_index = ignore_index
        self.free_index = free_index

        # if sem_sparse_backbone is not None:
        #     self.sem_sparse_backbone = MODELS.build(sem_sparse_backbone)

        self.conv_seg = self.build_conv_seg(channels=seg_channels, num_classes=num_classes, kernel_size=1)
        # if loss_focal is not None:
        #     self.loss_focal = MODELS.build(loss_focal)
        if loss_geo is not None:
            self.loss_geo = MODELS.build(loss_geo)
        if loss_sem is not None:
            self.loss_sem = MODELS.build(loss_sem)
        if loss_lovasz is not None:
            self.loss_lovasz = MODELS.build(loss_lovasz)

    def build_conv_seg(self, channels: int, num_classes: int, kernel_size: int) -> nn.Module:
        """Build Convolutional Segmentation Layers."""
        return nn.Conv1d(channels, num_classes, kernel_size=kernel_size)

    def sem_forward(self, fea: Tensor = None, coor: Tensor = None) -> Tensor:
        # B, C, N = fea.shape
        # fea = fea.permute(0, 2, 1).contiguous().view(-1, C)
        # coor = coor.view(-1, 4)
        # fea = self.sem_sparse_backbone(fea, coor).reshape(B, N, -1).permute(0, 2, 1)
        return self.conv_seg(fea)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        """Concat voxel-wise Groud Truth."""
        gt_semantic_segs = [data_sample.gt_pts_seg.voxel_label.long() for data_sample in batch_data_samples]

        return torch.stack(gt_semantic_segs, dim=0)

    def loss_by_feat(self, geo_logits: Tensor, sem_logits: Tensor, sc_query_grid_coor: Tensor, batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Compute semantic segmentation loss.

        Args:
            seg_logit (Tensor): Predicted per-point segmentation logits of
                shape [B, num_classes, N].
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        geo_label = torch.where(torch.logical_and(seg_label != self.free_index, seg_label != self.ignore_index), 1, seg_label)
        sem_label = seg_label[sc_query_grid_coor[:, :, 0], sc_query_grid_coor[:, :, 1], sc_query_grid_coor[:, :, 2], sc_query_grid_coor[:, :, 3]]
        # TODO error change dim
        loss = dict()
        # if hasattr(self, "loss_focal"):
        #     loss["loss_focal"] = self.loss_focal(seg_logit, seg_label, weight=self.class_weights, ignore_index=self.ignore_index)
        if hasattr(self, "loss_geo"):
            loss["loss_geo"] = self.loss_geo(geo_logits, geo_label, ignore_index=self.ignore_index)
        if hasattr(self, "loss_sem"):
            loss["loss_sem"] = self.loss_sem(sem_logits, sem_label, ignore_index=self.ignore_index)
        if hasattr(self, "loss_lovasz"):
            loss["loss_lovasz"] = self.loss_lovasz(torch.softmax(sem_logits, dim=1), sem_label, ignore_index=self.ignore_index)
        return loss

    def loss(
        self, geo_fea: Tensor, sem_fea: Tensor, sc_query_grid_coor: Tensor, batch_data_samples: SampleList, train_cfg: ConfigType = None
    ) -> Dict[str, Tensor]:
        """Forward function for training.

        Args:
            inputs (dict): Feature dict from backbone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            train_cfg (dict or :obj:`ConfigDict`): The training config.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        geo_logits = geo_fea
        sem_logits = self.sem_forward(sem_fea, sc_query_grid_coor)
        losses = self.loss_by_feat(geo_logits, sem_logits, sc_query_grid_coor, batch_data_samples)
        return losses

    def predict(self, geo_fea: Tensor, sem_fea: Tensor, sc_query_grid_coor: Tensor = None, batch_data_samples: SampleList = None) -> List[Tensor]:
        """Forward function for testing.

        Args:
            inputs (Tensor): Features from backone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg
                data samples.

        Returns:
            List[Tensor]: The segmentation prediction mask of each batch.
        """

        geo_logits = geo_fea
        sem_logits = self.sem_forward(sem_fea, sc_query_grid_coor)
        return geo_logits, sem_logits
