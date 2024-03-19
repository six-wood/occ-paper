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


class SegmentationHead(nn.Module):
    """
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    """

    def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
        super().__init__()

        # First convolution
        self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList([nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        self.conv2 = nn.ModuleList([nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        self.relu = nn.ReLU(inplace=True)

        # Convolution for output
        self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

    def forward(self, x_in):
        # Dimension exapension
        x_in = x_in[:, None, :, :, :]

        # Convolution to go from inplanes to planes features...
        x_in = self.relu(self.conv0(x_in))

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        x_in = self.conv_classes(x_in)

        return x_in


@MODELS.register_module()
class SscHead(BaseModule):
    """
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    """

    # TODO ADD SCPNet completion Head

    def __init__(
        self,
        loss_focal: ConfigType = None,
        loss_ce: ConfigType = None,
        conv_cfg: ConfigType = dict(type="Conv3d"),
        norm_cfg: ConfigType = dict(type="BN3d"),
        act_cfg: ConfigType = dict(type="ReLU"),
        init_cfg: OptMultiConfig = None,
        ignore_index: int = 255,
        free_index: int = 0,
    ):
        super(SscHead, self).__init__(init_cfg=init_cfg)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.ignore_index = ignore_index
        self.free_index = free_index

        self.sc_class = SegmentationHead(
            inplanes=1,
            planes=8,
            nbr_classes=20,
            dilations_conv_list=[1, 2, 3],
        )

        if loss_focal is not None:
            self.loss_focal = MODELS.build(loss_focal)
        if loss_ce is not None:
            self.loss_ce = MODELS.build(loss_ce)

    def forward(self, fea: Tensor = None) -> Tensor:
        return self.sc_class(fea)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        """Concat voxel-wise Groud Truth."""
        gt_semantic_segs = np.stack([data_sample.metainfo["voxel_label"] for data_sample in batch_data_samples], axis=0)
        return torch.from_numpy(gt_semantic_segs).long()

    def loss_by_feat(self, geo_logits: Tensor, batch_data_samples: SampleList) -> Dict[str, Tensor]:
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

        seg_label = self._stack_batch_gt(batch_data_samples).to(geo_logits.device)
        losses = dict()
        # if hasattr(self, "loss_focal"):
        #     loss["loss_focal"] = self.loss_focal(seg_logit, seg_label, weight=self.class_weights, ignore_index=self.ignore_index)
        losses["loss_sc"] = self.loss_ce(geo_logits, seg_label, ignore_index=self.ignore_index)
        return losses

    def loss(self, geo_fea: Tensor, batch_data_samples: SampleList, train_cfg: ConfigType = None) -> Dict[str, Tensor]:
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
        geo_logits = self.forward(geo_fea).permute(0, 1, 3, 4, 2)

        losses = self.loss_by_feat(geo_logits, batch_data_samples)
        return losses

    def predict(self, geo_fea: Tensor) -> List[Tensor]:
        """Forward function for testing.

        Args:
            inputs (Tensor): Features from backone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg
                data samples.

        Returns:
            List[Tensor]: The segmentation prediction mask of each batch.
        """

        geo_logits = self.forward(geo_fea[:, None, :, :, :]).permute(0, 1, 3, 4, 2)
        return geo_logits
