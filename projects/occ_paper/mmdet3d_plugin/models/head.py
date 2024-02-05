import torch.nn as nn
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from typing import List, Dict

import torch
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.utils.typing_utils import ConfigType
from mmdet3d.structures.det3d_data_sample import SampleList
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer


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
        loss_decode: ConfigType = dict(type="mmdet.CrossEntropyLoss", use_sigmoid=False, class_weight=None, loss_weight=1.0),
        ignore_index: int = 255,
    ):
        super(DenseSscHead, self).__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.loss_decode = MODELS.build(loss_decode)
        self.ignore_index = ignore_index
        # First convolution
        self.conv0 = build_conv_layer(self.conv_cfg, inplanes, planes, kernel_size=3, stride=1, padding=1)

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [build_conv_layer(self.conv_cfg, planes, planes, kernel_size=3, stride=1, padding=dil, dilation=dil) for dil in dilations_conv_list]
        )
        self.bn1 = nn.ModuleList([build_norm_layer(self.norm_cfg, planes)[1] for dil in dilations_conv_list])
        self.conv2 = nn.ModuleList(
            [build_conv_layer(self.conv_cfg, planes, planes, kernel_size=3, stride=1, padding=dil, dilation=dil) for dil in dilations_conv_list]
        )
        self.bn2 = nn.ModuleList([build_norm_layer(self.norm_cfg, planes)[1] for dil in dilations_conv_list])
        self.act = build_activation_layer(self.act_cfg)

        # Convolution for output
        self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

    def forward(self, x: Tensor):
        # Dimension exapension

        # Convolution to go from inplanes to planes features...
        x = self.act(self.conv0(x))

        y = self.bn2[0](self.conv2[0](self.act(self.bn1[0](self.conv1[0](x)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.act(self.bn1[i](self.conv1[i](x)))))
        x = self.act(y + x)  # modified

        x = self.conv_classes(x)

        return x.permute(0, 1, 3, 4, 2).contiguous()

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        """Concat voxel-wise Groud Truth."""
        gt_semantic_segs = [data_sample.gt_pts_seg.voxel_label.long() for data_sample in batch_data_samples]

        return torch.stack(gt_semantic_segs, dim=0)

    def loss_by_feat(self, seg_logit: Tensor, batch_data_samples: SampleList) -> Dict[str, Tensor]:
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
        loss = dict()
        loss["loss_sem_seg"] = self.loss_decode(seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss

    def loss(self, inputs: Tensor, batch_data_samples: SampleList, train_cfg: ConfigType) -> Dict[str, Tensor]:
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
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
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
