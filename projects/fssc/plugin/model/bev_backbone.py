import torch
import torch.nn as nn
from torch import Tensor
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from typing import Optional, Sequence, Tuple
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmcv.cnn import ConvModule, build_activation_layer, build_conv_layer, build_norm_layer


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
        act_cfg: ConfigType = dict(type="LeakyReLU"),
        init_cfg: OptMultiConfig = None,
        frozen: bool = False,
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
        if frozen:
            for param in self.parameters():
                param.requires_grad = False

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
class BevNet(BaseModule):
    def __init__(
        self,
        bev_input_dimensions: int = 32,
        bev_stem_channels: int = 32,
        bev_num_stages: int = 3,
        bev_stage_blocks: Sequence[int] = (4, 6, 3),
        bev_strides: Sequence[int] = (2, 2, 2),
        bev_dilations: Sequence[int] = (1, 1, 1),
        bev_encoder_out_channels: Sequence[int] = (48, 64, 80),
        bev_decoder_out_channels: Sequence[int] = (64, 48, 32),
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN"),
        act_cfg: ConfigType = dict(type="ReLU", inplace=True),
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(init_cfg=init_cfg)
        """
        SSCNet architecture
        :param N: number of classes to be predicted (i.e. 12 for NYUv2)
        """
        assert len(bev_encoder_out_channels) == len(bev_decoder_out_channels) == bev_num_stages, (
            "The length of encoder_out_channels, decoder_out_channels " "should be equal to num_stages"
        )
        # input downsample

        # model parameters
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        in_channel, stem_out = bev_input_dimensions, bev_stem_channels
        self.stem = self._make_conv_layer(in_channel, stem_out)
        # encoder_channels
        inplanes = stem_out
        e_out2 = bev_encoder_out_channels[0]
        e_out3 = bev_encoder_out_channels[1]
        e_out4 = bev_encoder_out_channels[2]

        # encode block
        self.res_layers = []
        for i, num_blocks in enumerate(bev_stage_blocks):
            stride = bev_strides[i]
            dilation = bev_dilations[i]
            planes = bev_encoder_out_channels[i]
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

        # decoder channels
        d_in1 = e_out4
        d_out1 = bev_decoder_out_channels[0]
        d_out2 = bev_decoder_out_channels[1]
        d_out3 = bev_decoder_out_channels[2]

        # decode block 1
        self.up_sample1 = nn.ConvTranspose2d(d_in1, d_in1, kernel_size=2, padding=0, stride=2, bias=False)
        self.conv_layer1 = self._make_conv_layer(d_in1 + e_out3, d_out1)

        # decode block 2
        self.up_sample2 = nn.ConvTranspose2d(d_out1, d_out1, kernel_size=2, padding=0, stride=2, bias=False)
        self.up_sample2_1 = nn.ConvTranspose2d(d_in1, d_in1, kernel_size=4, padding=0, stride=4, bias=False)
        self.conv_layer2 = self._make_conv_layer(d_out1 + d_in1 + e_out2, d_out2)

        # decode block 3
        self.up_sample3 = nn.ConvTranspose2d(d_out2, d_out2, kernel_size=2, padding=0, stride=2, bias=False)
        self.up_sample3_1 = nn.ConvTranspose2d(d_out1, d_out1, kernel_size=4, padding=0, stride=4, bias=False)
        self.up_sample3_2 = nn.ConvTranspose2d(d_in1, d_in1, kernel_size=8, padding=0, stride=8, bias=False)
        self.conv_layer3 = self._make_conv_layer(d_out2 + d_out1 + d_in1 + stem_out, d_out3)

    def _make_conv_layer(self, in_channels: int, out_channels: int) -> None:  # two conv blocks in beginning
        return nn.Sequential(
            ConvModule(in_channels, out_channels, kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
            ConvModule(out_channels, out_channels, kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
        )

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
            )

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
        )
        inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(inplanes=inplanes, planes=planes, stride=1, dilation=dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            )  # add the residual blocks
        return nn.Sequential(*layers)

    def forward(self, bev_map: Tensor = None):
        # Encoder
        x = self.stem(bev_map)  # [bs, 32, 256, 256]
        outs = [x]
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            outs.append(x)

        # Decoder1 out_1/4
        dec1 = self.up_sample1(outs[-1])  # [bs, 80, 64, 64]
        dec1 = torch.cat([dec1, outs[-2]], dim=1)  # [bs, 80+64, 64, 64]
        dec1 = self.conv_layer1(dec1)  # [bs, 64, 64, 64]

        # Decoder2 out_1/2
        dec2 = self.up_sample2(dec1)  # [bs, 64, 128, 128]
        fuse2_1 = self.up_sample2_1(outs[-1])  # [bs, 80, 128, 128]
        dec2 = torch.cat([dec2, outs[-3], fuse2_1], dim=1)  # [bs, 64+48+80, 128, 128]
        dec2 = self.conv_layer2(dec2)  # [bs, 48, 128, 128]

        # Decoder3 out_1
        dec3 = self.up_sample3(dec2)  # [bs, 48, 256, 256]
        fuse3_1 = self.up_sample3_1(dec1)  # [bs, 64, 256, 256]
        fuse3_2 = self.up_sample3_2(outs[-1])  # [bs, 80, 256, 256]
        dec3 = torch.cat([dec3, outs[-4], fuse3_1, fuse3_2], dim=1)  # [bs, 48+32+64+80, 256, 256]
        dec3 = self.conv_layer3(dec3)  # [bs, 32, 256, 256]
        # dec3 = self.atten_block3(dec3, bev_map)

        return dec3
