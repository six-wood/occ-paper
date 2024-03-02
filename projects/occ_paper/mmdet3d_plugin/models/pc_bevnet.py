import torch
import torch.nn as nn
from typing import Sequence
from torch import Tensor
from .moudle import ResBlock
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmdet3d.utils import ConfigType, OptConfigType
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer, ConvModule
from .moudle import make_res_layer


@MODELS.register_module()
class BevNet(BaseModule):
    def __init__(
        self,
        input_dimensions=32,
        stem_channels: int = 32,
        num_stages: int = 4,
        stage_blocks: Sequence[int] = (2, 2, 2, 2),
        strides: Sequence[int] = (1, 2, 2, 2),
        dilations: Sequence[int] = (1, 1, 1, 1),
        encoder_out_channels: Sequence[int] = (48, 64, 80, 96),
        decoder_out_channels: Sequence[int] = (80, 64, 48, 32),
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN"),
        act_cfg: ConfigType = dict(type="LeakyReLU"),
    ):
        super().__init__()
        """
        SSCNet architecture
        :param N: number of classes to be predicted (i.e. 12 for NYUv2)
        """
        assert len(encoder_out_channels) == len(decoder_out_channels) == num_stages, (
            "The length of encoder_out_channels, decoder_out_channels " "should be equal to num_stages"
        )
        # input downsample

        # model parameters
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        in_channel, stem_out = input_dimensions, stem_channels
        self.stem = self._make_stem_layer(in_channel, stem_out)
        # encoder_channels
        inplanes = stem_out
        e_out1 = encoder_out_channels[0]
        e_out2 = encoder_out_channels[1]
        e_out3 = encoder_out_channels[2]
        e_out4 = encoder_out_channels[3]

        # encode block
        self.res_layers = []
        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = encoder_out_channels[i]
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

        # decoder channels
        d_in1 = e_out4
        d_out1 = decoder_out_channels[0]
        d_out2 = decoder_out_channels[1]
        d_out3 = decoder_out_channels[2]
        d_out4 = decoder_out_channels[3]

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
        self.conv_layer3 = self._make_conv_layer(d_out2 + d_out1 + d_in1 + e_out1, d_out3)

        # out block
        self.conv_layer4 = self._make_conv_layer(d_out3 + d_out4 + stem_out, d_out4)

    def _make_stem_layer(self, in_channels: int, out_channels: int) -> None:  # tree conv blocks in beginning
        return nn.Sequential(
            ConvModule(in_channels, out_channels, kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
            ConvModule(out_channels, out_channels, kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
            ConvModule(out_channels, out_channels, kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
        )

    def _make_conv_layer(self, in_channels: int, out_channels: int) -> None:  # two conv blocks in beginning
        return nn.Sequential(
            ConvModule(in_channels, out_channels, kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
            ConvModule(out_channels, out_channels, kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
        )

    def forward(self, bev_map: Tensor = None):
        # Encoder
        x = self.stem(bev_map)  # [bs, 32, 256, 256]
        outs = [x]
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            outs.append(x)

        # Decoder1 out_1/4
        dec1 = self.up_sample1(outs[-1])  # [bs, 96, 64, 64]
        dec1 = torch.cat([dec1, outs[-2]], dim=1)  # [bs, 96+80, 64, 64]
        dec1 = self.conv_layer1(dec1)  # [bs, 64, 64, 64]

        # Decoder2 out_1/2
        dec2 = self.up_sample2(dec1)  # [bs, 80, 128, 128]
        fuse2_1 = self.up_sample2_1(outs[-1])  # [bs, 96, 128, 128]
        dec2 = torch.cat([dec2, outs[-3], fuse2_1], dim=1)  # [bs, 80+64+96, 128, 128]
        dec2 = self.conv_layer2(dec2)  # [bs, 64, 128, 128]

        # Decoder3 out_1
        dec3 = self.up_sample3(dec2)  # [bs, 64, 256, 256]
        fuse3_1 = self.up_sample3_1(dec1)  # [bs, 80, 256, 256]
        fuse3_2 = self.up_sample3_2(outs[-1])  # [bs, 96, 256, 256]
        dec3 = torch.cat([dec3, outs[-4], fuse3_1, fuse3_2], dim=1)  # [bs, 64+48+80+96, 256, 256]
        dec3 = self.conv_layer3(dec3)  # [bs, 48, 256, 256]
        # dec3 = self.atten_block3(dec3, bev_map)
        dec4 = torch.cat([dec3, outs[-5], bev_map], dim=1)  # [bs, 32+32+48, 256, 256]
        dec4 = self.conv_layer4(dec4)
        return dec4
