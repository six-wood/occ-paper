import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Tuple
from torch import Tensor
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule, build_activation_layer, build_conv_layer, build_norm_layer
from mmdet3d.utils import ConfigType, OptConfigType
from mmengine.model import BaseModule
from .utils import make_res_layer


@MODELS.register_module()
class RangeNet(BaseModule):
    def __init__(
        self,
        in_channels: int = 5,
        stem_channels: int = 128,
        num_stages: int = 4,
        stage_blocks: Sequence[int] = (3, 4, 6, 3),
        out_channels: Sequence[int] = (128, 128, 128, 128),
        strides: Sequence[int] = (1, 2, 2, 2),
        dilations: Sequence[int] = (1, 1, 1, 1),
        fuse_channels: Sequence[int] = (256, 128),
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN"),
        act_cfg: ConfigType = dict(type="LeakyReLU"),
        init_cfg=None,
    ) -> None:
        super(RangeNet, self).__init__(init_cfg)

        assert len(stage_blocks) == len(out_channels) == len(strides) == len(dilations) == num_stages, (
            "The length of stage_blocks, out_channels, strides and " "dilations should be equal to num_stages"
        )
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self._make_stem_layer(in_channels, stem_channels)

        inplanes = stem_channels
        self.res_layers = []
        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = out_channels[i]
            res_layer = make_res_layer(
                inplanes=inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            inplanes = planes
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        in_channels = stem_channels + sum(out_channels)
        self.fuse_layers = []
        for i, fuse_channel in enumerate(fuse_channels):
            fuse_layer = ConvModule(
                in_channels, fuse_channel, kernel_size=3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
            )  # conv block that bundles conv/norm/activation layers.
            in_channels = fuse_channel
            layer_name = f"fuse_layer{i + 1}"
            self.add_module(layer_name, fuse_layer)
            self.fuse_layers.append(layer_name)

    def _make_stem_layer(self, in_channels: int, out_channels: int) -> None:  # tree conv blocks in beginning
        self.stem = nn.Sequential(
            ConvModule(in_channels, out_channels, kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
            ConvModule(out_channels, out_channels, kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
            ConvModule(out_channels, out_channels, kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
        )  # conv block that bundles conv/norm/activation layers.

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x = self.stem(x)
        outs = [x]

        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            outs.append(x)

        # TODO: move the following operation into neck.
        for i in range(len(outs)):
            if outs[i].shape != outs[0].shape:
                outs[i] = F.interpolate(outs[i], size=outs[0].size()[2:], mode="bilinear", align_corners=True)  # interpolate to match the dimensions

        outs[0] = torch.cat(outs, dim=1)  # concatenate the outputs of the residual blocks

        for layer_name in self.fuse_layers:
            fuse_layer = getattr(self, layer_name)
            outs[0] = fuse_layer(outs[0])
        return tuple(outs)
