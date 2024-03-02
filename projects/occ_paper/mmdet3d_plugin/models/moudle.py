import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from mmengine.model import BaseModule
from spconv.pytorch import SparseConvTensor
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmseg.models.decode_heads.aspp_head import ASPPModule


def replace_feature(out: SparseConvTensor, new_features: SparseConvTensor) -> SparseConvTensor:
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


class ResBlock(BaseModule):
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
        super(ResBlock, self).__init__(init_cfg)

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


def make_res_layer(
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
        ResBlock(
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
            ResBlock(inplanes=inplanes, planes=planes, stride=1, dilation=dilation, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )  # add the residual blocks
    return nn.Sequential(*layers)


class ASPP3D(BaseModule):
    def __init__(self, in_channels, mid_channels, out_channels, dilations=(1, 2, 3), conv_cfg=None, norm_cfg=None, act_cfg=None, **kwargs):
        super().__init__()
        # First convolution
        self.conv0 = build_conv_layer(conv_cfg, in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.bn0 = build_norm_layer(norm_cfg, mid_channels)[1]

        # ASPP Block
        self.conv_list = dilations
        num = len(dilations)
        self.conv1 = nn.ModuleList(
            [build_conv_layer(conv_cfg, mid_channels, mid_channels, kernel_size=3, stride=1, padding=dil, dilation=dil) for dil in dilations]
        )
        self.bn1 = nn.ModuleList([build_norm_layer(norm_cfg, mid_channels)[1] for dil in dilations])

        self.conv2 = nn.ModuleList(
            [build_conv_layer(conv_cfg, mid_channels, mid_channels, kernel_size=3, stride=1, padding=dil, dilation=dil) for dil in dilations]
        )
        self.bn2 = nn.ModuleList([build_norm_layer(norm_cfg, out_channels)[1] for dil in dilations])

        # Convolution for output
        self.conv3 = build_conv_layer(conv_cfg, mid_channels * (num + 1), out_channels, kernel_size=3, padding=1, stride=1)
        self.bn3 = build_norm_layer(norm_cfg, out_channels)[1]

        self.act = build_activation_layer(act_cfg)

    def forward(self, fea: Tensor, coors: Tensor) -> Tensor:
        spatial_shape = coors.max(0)[0][1:] + 1
        batch_size = int(coors[-1, 0]) + 1
        x = SparseConvTensor(fea, coors, spatial_shape, batch_size)

        # Convolution to go from inplanes to planes features...
        x = self.conv0(x)
        x = replace_feature(x, self.bn0(x.features))
        x = replace_feature(x, self.act(x.features))
        y = [x.features]
        for i in range(len(self.conv_list)):
            x_ = self.conv1[i](x)
            x_ = replace_feature(x_, self.bn1[i](x_.features))
            x_ = replace_feature(x_, self.act(x_.features))
            x_ = self.conv2[i](x_)
            x_ = replace_feature(x_, self.bn2[i](x_.features))
            y.append(x_.features)

        x = replace_feature(x, torch.cat(y, dim=1))
        x = self.conv3(x)
        x = replace_feature(x, self.bn3(x.features))
        x = replace_feature(x, self.act(x.features))

        return x.features


def compute_visibility_mask(
    center: list = [0, 0, 0],
    pc_range: list = [0, -25.6, -2.0, 51.2, 25.6, 4.4],
    voxel_size: list = [0.2, 0.2, 0.2],
    fov: list = [-25.0, 3.0],
) -> np.ndarray:
    # 计算网格大小
    pc_range = np.array(pc_range)
    voxel_size = np.array(voxel_size)
    fov = np.array(fov)
    grid_size = np.round((pc_range[3:] - pc_range[:3]) / voxel_size).astype(np.int32)

    # 确定每个轴的范围
    x_range = np.linspace(pc_range[0] + voxel_size[0] / 2, pc_range[3] - voxel_size[0] / 2, grid_size[0])
    y_range = np.linspace(pc_range[1] + voxel_size[1] / 2, pc_range[4] - voxel_size[1] / 2, grid_size[1])
    z_range = np.linspace(pc_range[2] + voxel_size[2] / 2, pc_range[5] - voxel_size[2] / 2, grid_size[2])

    # 生成三维网格
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing="ij")

    # 调整网格以反映中心点的偏移
    xx -= center[0]
    yy -= center[1]
    zz -= center[2]

    # 计算每个点的俯仰角
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    pitch_angles = np.arcsin(zz / r)

    # 转换为度
    pitch_angles_degrees = np.degrees(pitch_angles)

    # 确定每个体素是否在视场范围内
    visibility_mask = (pitch_angles_degrees >= fov[0]) & (pitch_angles_degrees <= fov[1])

    return visibility_mask
