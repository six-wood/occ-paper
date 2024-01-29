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
from .moudle import BasicBlock


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
            res_layer = self.make_res_layer(
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
            build_conv_layer(self.conv_cfg, in_channels, out_channels // 2, kernel_size=3, padding=1, bias=False),
            build_norm_layer(self.norm_cfg, out_channels // 2)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(self.conv_cfg, out_channels // 2, out_channels, kernel_size=3, padding=1, bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(self.conv_cfg, out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg),
        )  # conv block that bundles conv/norm/activation layers.

    def make_res_layer(
        self,
        inplanes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        dilation: int,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN"),
        act_cfg: ConfigType = dict(type="LeakyReLU"),
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes:  # downsample to match the dimensions
            downsample = nn.Sequential(
                build_conv_layer(conv_cfg, inplanes, planes, kernel_size=1, stride=stride, bias=False), build_norm_layer(norm_cfg, planes)[1]
            )  # configure the downsample layer

        layers = []
        layers.append(
            BasicBlock(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        )  # add the first residual block
        inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(inplanes=inplanes, planes=planes, stride=1, dilation=dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            )  # add the residual blocks
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x = self.stem(x)
        outs = [x]
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            outs.append(x)

        # TODO: move the following operation into neck.
        rpn_outs = []
        for i in range(len(outs)):
            if outs[i].shape != outs[0].shape:
                rpn_outs.append(
                    F.interpolate(outs[i], size=outs[0].size()[2:], mode="bilinear", align_corners=True)
                )  # interpolate to match the dimensions
            else:
                rpn_outs.append(outs[i])

        outs[0] = torch.cat(rpn_outs, dim=1)  # concatenate the outputs of the residual blocks

        for layer_name in self.fuse_layers:
            fuse_layer = getattr(self, layer_name)
            outs[0] = fuse_layer(outs[0])
        return tuple(outs)