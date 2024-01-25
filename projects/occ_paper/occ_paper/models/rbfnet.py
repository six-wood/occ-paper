import os
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import numpy as np
from torch import Tensor
from torch._tensor import Tensor
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmdet3d.structures import Det3DDataSample
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmcv.cnn import ConvModule, build_activation_layer, build_conv_layer, build_norm_layer
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule


class CrossChannelAttentionModule(BaseModule):
    def __init__(self, num_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        b, c, _, _ = x.size()
        att = self.avg_pool(y).view(b, c)
        att = self.fc(att).view(b, c, 1, 1)
        return x * att.expand_as(x)


class BasicBlock(BaseModule):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN"),
        act_cfg: ConfigType = dict(type="ReLU"),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super(BasicBlock, self).__init__(init_cfg)

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(conv_cfg, inplanes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample

    @property
    def norm1(self) -> nn.Module:
        """nn.Module: normalization layer after the first convolution layer."""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self) -> nn.Module:
        """nn.Module: normalization layer after the second convolution layer."""
        return getattr(self, self.norm2_name)

    def forward(self, x: Tensor) -> Tensor:
        """
        ResBlock: two conv layers with a residual connection.
        """
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


@MODELS.register_module()
class BEVNet(BaseModule):
    def __init__(
        self,
        input_dimensions=None,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN"),
        act_cfg: ConfigType = dict(type="ReLU"),
    ):
        super().__init__()
        """
        SSCNet architecture
        :param N: number of classes to be predicted (i.e. 12 for NYUv2)
        """

        # model parameters
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        f = input_dimensions
        # encoder_channels
        e_in1, e_out1 = int(f), int(f)
        e_out2 = int(f * 1.5)
        e_out3 = int(f * 2)
        e_out4 = int(f * 2.5)

        # encode block
        self.Encoder_block1 = self._make_conv_layer(e_in1, e_out1)
        self.Encoder_block2 = self._make_encoder_layer(e_out1, e_out2)
        self.Encoder_block3 = self._make_encoder_layer(e_out2, e_out3)
        self.Encoder_block4 = self._make_encoder_layer(e_out3, e_out4)

        # decoder channels
        d_in1 = e_out4
        d_out1 = int(f * 2)
        d_out2 = int(f * 1.5)
        d_out3 = int(f)

        # decode block 1
        self.up_sample1 = nn.ConvTranspose2d(d_in1, d_in1, kernel_size=2, padding=0, stride=2, bias=False)
        self.conv_layer1 = self._make_conv_layer(d_in1 + e_out3, d_out1)
        self.atten_block1 = CrossChannelAttentionModule(d_out1)

        # decode block 2
        self.up_sample2 = nn.ConvTranspose2d(d_out1, d_out1, kernel_size=2, padding=0, stride=2, bias=False)
        self.up_sample2_1 = nn.ConvTranspose2d(d_in1, d_in1, kernel_size=4, padding=0, stride=4, bias=False)
        self.conv_layer2 = self._make_conv_layer(d_out1 + d_in1 + e_out2, int(f * 1.5))
        self.atten_block2 = CrossChannelAttentionModule(int(f * 1.5))

        # decode block 3
        self.up_sample3 = nn.ConvTranspose2d(d_out2, d_out2, kernel_size=2, padding=0, stride=2, bias=False)
        self.up_sample3_1 = nn.ConvTranspose2d(d_out1, d_out1, kernel_size=4, padding=0, stride=4, bias=False)
        self.up_sample3_2 = nn.ConvTranspose2d(d_in1, d_in1, kernel_size=8, padding=0, stride=8, bias=False)
        self.conv_layer3 = self._make_conv_layer(d_out2 + d_out1 + d_in1 + e_out1, d_out3)
        self.atten_block3 = CrossChannelAttentionModule(d_out3)

    def _make_conv_layer(self, in_channels: int, out_channels: int) -> None:  # two conv blocks in beginning
        return nn.Sequential(
            build_conv_layer(self.conv_cfg, in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_activation_layer(self.act_cfg),
            build_conv_layer(self.conv_cfg, out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_activation_layer(self.act_cfg),
        )

    def _make_encoder_layer(self, in_channels: int, out_channels: int) -> None:  # create encoder blocks
        return nn.Sequential(
            nn.MaxPool2d(2),
            build_conv_layer(self.conv_cfg, in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_activation_layer(self.act_cfg),
            build_conv_layer(self.conv_cfg, out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_activation_layer(self.act_cfg),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.Encoder_block1(x)  # [bs, 32, 256, 256]
        enc2 = self.Encoder_block2(enc1)  # [bs, 48, 128, 128]
        enc3 = self.Encoder_block3(enc2)  # [bs, 64, 64, 64]
        mec0 = self.Encoder_block4(enc3)  # [bs, 80, 32, 32]

        # Decoder1 out_1/4
        dec1 = self.up_sample1(mec0)  # [bs, 80, 64, 64]
        dec1 = torch.cat([dec1, enc3], dim=1)  # [bs, 80+64, 64, 64]
        dec1 = self.conv_layer1(dec1)  # [bs, 64, 64, 64]
        dec1 = self.atten_block1(dec1, enc3)  # [bs, 64, 64, 64]

        # Decoder2 out_1/2
        dec2 = self.up_sample2(dec1)  # [bs, 64, 128, 128]
        fuse2_1 = self.up_sample2_1(mec0)  # [bs, 80, 128, 128]
        dec2 = torch.cat([dec2, enc2, fuse2_1], dim=1)  # [bs, 64+48+80, 128, 128]
        dec2 = self.conv_layer2(dec2)  # [bs, 48, 128, 128]
        dec2 = self.atten_block2(dec2, enc2)  # [bs, 48, 128, 128]

        # Decoder3 out_1
        dec3 = self.up_sample3(dec2)  # [bs, 48, 256, 256]
        fuse3_1 = self.up_sample3_1(dec1)  # [bs, 64, 256, 256]
        fuse3_2 = self.up_sample3_2(mec0)  # [bs, 80, 256, 256]
        dec3 = torch.cat([dec3, enc1, fuse3_1, fuse3_2], dim=1)  # [bs, 48+32+64+80, 256, 256]
        dec3 = self.conv_layer3(dec3)  # [bs, 32, 256, 256]
        out_2D = self.atten_block3(dec3, x)  # [bs, 32, 256, 256]

        return out_2D


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
        for i in range(len(outs)):
            if outs[i].shape != outs[0].shape:
                outs[i] = F.interpolate(outs[i], size=outs[0].size()[2:], mode="bilinear", align_corners=True)  # interpolate to match the dimensions

        outs[0] = torch.cat(outs, dim=1)  # concatenate the outputs of the residual blocks

        for layer_name in self.fuse_layers:
            fuse_layer = getattr(self, layer_name)
            outs[0] = fuse_layer(outs[0])
        return tuple(outs)
