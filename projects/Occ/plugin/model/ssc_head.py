import torch.nn as nn
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from typing import List, Dict

import torch
import numpy as np
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.utils.typing_utils import ConfigType, OptMultiConfig
from mmdet3d.structures.det3d_data_sample import SampleList

from mmcv.cnn import ConvModule


@MODELS.register_module()
class SscHead(BaseModule):
    r"""MinkUNet decoder head with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

    Args:
        channels (int): The input channel of conv_seg.
        num_classes (int): Number of classes.
    """

    def __init__(
        self,
        channels: int,
        num_classes: int,
        dropout_ratio: float = 0.5,
        conv_cfg: ConfigType = dict(type="Conv1d"),
        norm_cfg: ConfigType = dict(type="BN1d"),
        act_cfg: ConfigType = dict(type="ReLU"),
        loss_ce: ConfigType = dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0,
        ),
        loss_lovasz: ConfigType = dict(
            type="LovaszLoss",
            loss_weight=1.0,
            reduction="none",
        ),
        conv_seg_kernel_size: int = 1,
        ignore_index: int = 255,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super(SscHead, self).__init__(init_cfg=init_cfg)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.loss_ce = MODELS.build(loss_ce)
        self.loss_lovasz = MODELS.build(loss_lovasz)
        self.ignore_index = ignore_index

        self.cls_seg = self.build_cls_seg(channels=channels, num_classes=num_classes, kernel_size=conv_seg_kernel_size)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
        else:
            self.dropout = None

    def build_cls_seg(self, channels: int, num_classes: int, kernel_size: int) -> nn.Module:
        """Build Convolutional Segmentation Layers."""
        return nn.Linear(channels, num_classes)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        """Concat voxel-wise Groud Truth."""
        ssc_semantic_segs = np.stack([data_sample.metainfo["voxel_label"] for data_sample in batch_data_samples], axis=0)
        ssc_semantic_segs = torch.from_numpy(ssc_semantic_segs).to(torch.long)
        return ssc_semantic_segs

    def loss_by_feat(self, seg_logit: Tensor, coors: Tensor, batch_data_samples: SampleList) -> dict:
        """Compute semantic segmentation loss.

        Args:
            coors: b z y x
        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        ssc_true = self._stack_batch_gt(batch_data_samples).to(seg_logit.device)[
            coors[:, 0],
            coors[:, 3],
            coors[:, 2],
            coors[:, 1],
        ]

        loss = dict()

        loss["loss_ce"] = self.loss_ce(seg_logit, ssc_true, ignore_index=self.ignore_index)
        loss["loss_lovasz"] = self.loss_lovasz(seg_logit, ssc_true, ignore_index=self.ignore_index)
        return loss

    def loss(self, inputs: dict, batch_data_samples: SampleList, train_cfg: ConfigType) -> Dict[str, Tensor]:
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
        sem_fea = inputs["sem_fea"]
        coors = inputs["coors"]
        seg_logits = self.forward(sem_fea)
        losses = self.loss_by_feat(seg_logits, coors, batch_data_samples)
        return losses

    def predict(self, inputs: Tensor, batch_data_samples: SampleList) -> List[Tensor]:
        """Forward function for testing.

        Args:
            inputs (Tensor): Features from backone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg
                data samples.

        Returns:
            List[Tensor]: The segmentation prediction mask of each batch.
        """
        seg_logits = self.forward(inputs)

        return seg_logits

    def forward(self, sem_fea: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Features from backbone.

        Returns:
            Tensor: Segmentation map of shape [N, C].
                Note that output contains all points from each batch.
        """
        sem_logits = self.cls_seg(sem_fea)
        return sem_logits
