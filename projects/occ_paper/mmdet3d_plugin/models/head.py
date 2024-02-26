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


@MODELS.register_module()
class DenseSscHead(BaseModule):
    """
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    """

    # TODO ADD SCPNet completion Head

    def __init__(
        self,
        inplanes,
        planes,
        nbr_classes,
        dilations_conv_list,
        conv_cfg: ConfigType = dict(type="Conv3d"),
        norm_cfg: ConfigType = dict(type="BN3d"),
        act_cfg: ConfigType = dict(type="ReLU"),
        # loss_focal: ConfigType = None,
        loss_geo: ConfigType = None,
        loss_sem: ConfigType = None,
        loss_lovasz: ConfigType = None,
        ignore_index: int = 255,
    ):
        super(DenseSscHead, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.ignore_index = ignore_index
        # if loss_focal is not None:
        #     self.loss_focal = MODELS.build(loss_focal)
        if loss_geo is not None:
            self.loss_geo = MODELS.build(loss_geo)
        if loss_sem is not None:
            self.loss_sem = MODELS.build(loss_sem)
        if loss_lovasz is not None:
            self.loss_lovasz = MODELS.build(loss_lovasz)

    def sem_forward(self, x: Tensor):
        pass

    def geo_forward(self, x: Tensor):
        return x  # B, H, W, Z, C

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        """Concat voxel-wise Groud Truth."""
        gt_semantic_segs = [data_sample.gt_pts_seg.voxel_label.long() for data_sample in batch_data_samples]

        return torch.stack(gt_semantic_segs, dim=0)

    def loss_by_feat(self, geo_logits: Tensor, sem_logits: Tensor, batch_data_samples: SampleList, sc_query) -> Dict[str, Tensor]:
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
        B = seg_label.shape[0]
        sem_label = seg_label.view(B, -1)
        sem_label = sem_label.gather(1, sc_query)
        # TODO error change dim
        loss = dict()
        # if hasattr(self, "loss_focal"):
        #     loss["loss_focal"] = self.loss_focal(seg_logit, seg_label, weight=self.class_weights, ignore_index=self.ignore_index)
        if hasattr(self, "loss_geo"):
            loss["loss_geo"] = self.loss_geo(geo_logits, seg_label, ignore_index=self.ignore_index)
        if hasattr(self, "loss_sem"):
            loss["loss_sem"] = self.loss_sem(sem_logits, sem_label, ignore_index=self.ignore_index)
        if hasattr(self, "loss_lovasz"):
            loss["loss_lovasz"] = self.loss_lovasz(torch.softmax(sem_logits, dim=1), sem_label, ignore_index=self.ignore_index)
        return loss

    def loss(
        self, geo_fea: Tensor, sem_fea: Tensor, sc_query: Tensor, batch_data_samples: SampleList, train_cfg: ConfigType = None
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
        sem_logits = self.sem_forward(sem_fea)
        geo_logits = self.geo_forward(geo_fea)
        losses = self.loss_by_feat(geo_logits, sem_logits, batch_data_samples, sc_query)
        return losses

    def predict(self, geo_fea: Tensor, sem_fea: Tensor, sc_query: Tensor = None, batch_data_samples: SampleList = None) -> List[Tensor]:
        """Forward function for testing.

        Args:
            inputs (Tensor): Features from backone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg
                data samples.

        Returns:
            List[Tensor]: The segmentation prediction mask of each batch.
        """

        geo_logits = self.geo_forward(geo_fea)
        sem_logits = self.sem_forward(sem_fea)
        return geo_logits, sem_logits
