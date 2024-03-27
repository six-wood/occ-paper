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
from spconv.pytorch import SparseConvTensor

from mmdet3d.models.layers.sparse_block import replace_feature
from mmcv.cnn import build_conv_layer, build_norm_layer


class SegmentationHead(nn.Module):
    """
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    """

    def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
        super().__init__()

        # First convolution
        self.conv0 = build_conv_layer(dict(type="SubMConv3d"), inplanes, planes, kernel_size=3, padding=1, stride=1)

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [
                build_conv_layer(dict(type="SubMConv3d"), planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False)
                for dil in dilations_conv_list
            ]
        )
        self.bn1 = nn.ModuleList([build_norm_layer(dict(type="BN1d"), planes)[1] for dil in dilations_conv_list])
        self.conv2 = nn.ModuleList(
            [
                build_conv_layer(dict(type="SubMConv3d"), planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False)
                for dil in dilations_conv_list
            ]
        )
        self.bn2 = nn.ModuleList([build_norm_layer(dict(type="BN1d"), planes)[1] for dil in dilations_conv_list])
        self.relu = nn.ReLU(inplace=True)

        # Convolution for output
        self.conv_classes = build_conv_layer(dict(type="SubMConv3d"), planes, nbr_classes, kernel_size=3, padding=1, stride=1)

    def forward(self, feas: Tensor, coors: Tensor):
        # Convolution to go from inplanes to planes features...
        spatial_shape = coors.max(0)[0][1:] + 1
        batch_size = int(coors[-1, 0]) + 1
        x = SparseConvTensor(feas, coors, spatial_shape, batch_size)

        x = self.conv0(x)
        x = replace_feature(x, self.relu(x.features))

        y = self.conv1[0](x)
        y = replace_feature(y, self.relu(self.bn1[0](y.features)))
        y = self.conv2[0](y)
        y = replace_feature(y, self.bn2[0](y.features))

        for i in range(1, len(self.conv_list)):
            y_ = self.conv1[i](x)
            y_ = replace_feature(y_, self.relu(self.bn1[i](y_.features)))
            y_ = self.conv2[i](y_)
            y_ = replace_feature(y_, self.bn2[i](y_.features))
            y = replace_feature(y, y.features + y_.features)

        x = replace_feature(x, self.relu(y.features + x.features))

        x = self.conv_classes(x)

        return x.features


@MODELS.register_module()
class SscHead(BaseModule):
    """
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    """

    # TODO ADD SCPNet completion Head

    def __init__(
        self,
        # out_channels: int = 96,
        num_classes: int = 20,
        voxel_net: ConfigType = None,
        loss_focal: ConfigType = None,
        loss_ce: ConfigType = None,
        loss_lovasz: ConfigType = None,
        init_cfg: OptMultiConfig = None,
        ignore_index: int = 255,
        free_index: int = 0,
    ):
        super(SscHead, self).__init__(init_cfg=init_cfg)

        self.ignore_index = ignore_index
        self.free_index = free_index

        # self.class_seg = nn.Linear(out_channels, num_classes)
        self.class_seg = SegmentationHead(16, 32, num_classes, [1, 2, 3])

        # if voxel_net is not None:
        #     self.voxel_net = MODELS.build(voxel_net)

        if loss_focal is not None:
            self.loss_focal = MODELS.build(loss_focal)
        if loss_ce is not None:
            self.loss_ce = MODELS.build(loss_ce)
        if loss_lovasz is not None:
            self.loss_lovasz = MODELS.build(loss_lovasz)

    def forward(self, x: SparseConvTensor) -> Tensor:
        # fea = self.voxel_net(x.features, x.indices)
        # logits = self.class_seg(fea)
        logits = self.class_seg(x.features, x.indices)
        return logits

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        """Concat voxel-wise Groud Truth."""
        gt_semantic_segs = np.stack([data_sample.metainfo["voxel_label"] for data_sample in batch_data_samples], axis=0)
        return torch.from_numpy(gt_semantic_segs).long()

    def loss_by_feat(self, geo_logits: Tensor, coors: Tensor, batch_data_samples: SampleList) -> Dict[str, Tensor]:
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

        ssc_label = self._stack_batch_gt(batch_data_samples).to(geo_logits.device)[coors[:, 0], coors[:, 3], coors[:, 2], coors[:, 1]]
        losses = dict()
        # if hasattr(self, "loss_focal"):
        #     loss["loss_focal"] = self.loss_focal(seg_logit, seg_label, weight=self.class_weights, ignore_index=self.ignore_index)
        losses["loss_ce"] = self.loss_ce(geo_logits, ssc_label, ignore_index=self.ignore_index)
        losses["loss_lovasz"] = self.loss_lovasz(geo_logits, ssc_label, ignore_index=self.ignore_index)
        return losses

    def loss(self, x: SparseConvTensor, batch_data_samples: SampleList, train_cfg: ConfigType = None) -> Dict[str, Tensor]:
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

        geo_logits = self.forward(x)

        losses = self.loss_by_feat(geo_logits, x.indices, batch_data_samples)
        return losses

    def predict(self, x: SparseConvTensor) -> List[Tensor]:
        """Forward function for testing.

        Args:
            inputs (Tensor): Features from backone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg
                data samples.

        Returns:
            List[Tensor]: The segmentation prediction mask of each batch.
        """

        geo_logits = self.forward(x)
        return geo_logits
