import torch
import torch.nn as nn
import numpy as np
from typing import Sequence
from torch import Tensor
from .moudle import BasicBlock
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmdet3d.utils import ConfigType, OptConfigType
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer


@MODELS.register_module()
class FusionNet(BaseModule):
    def __init__(
        self,
        grid_shape: Sequence[int] = [256, 256, 32],
        range_shape: Sequence[int] = [64, 1024],
        pc_range: Sequence[float] = [0, -25.6, -2.0, 51.2, 25.6, 4.4],
        range_fov: Sequence[float] = [-90, -25, 90, 2],
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN"),
        act_cfg: ConfigType = dict(type="LeakyReLU"),
    ):
        super().__init__()
        voxel_grid_shape = np.array(grid_shape)
        voxel_size = (np.array(pc_range[3:]) - np.array(pc_range[:3])) / voxel_grid_shape
        xx, yy, zz = np.meshgrid(
            np.linspace(pc_range[0] + voxel_size[0] / 2, pc_range[3] - voxel_size[0] / 2, voxel_grid_shape[0]),
            np.linspace(pc_range[1] + voxel_size[1] / 2, pc_range[4] - voxel_size[1] / 2, voxel_grid_shape[1]),
            np.linspace(pc_range[2] + voxel_size[2] / 2, pc_range[5] - voxel_size[2] / 2, voxel_grid_shape[2]),
            indexing="ij",
        )
        r_voxel = np.sqrt(xx**2 + yy**2 + zz**2)
        yaw_voxel = -np.arctan2(yy, xx)
        pit_voxel = np.arcsin(zz / r_voxel)

        fov_left = range_fov[0] * np.pi / 180
        fov_down = range_fov[1] * np.pi / 180
        fov_x = (range_fov[2] - range_fov[0]) * np.pi / 180
        fov_y = (range_fov[3] - range_fov[1]) * np.pi / 180

        voxel_x = ((yaw_voxel + abs(fov_left)) / fov_x) * range_shape[1]
        voxel_y = (1.0 - (pit_voxel + abs(fov_down)) / fov_y) * range_shape[0]

        mask = (voxel_x >= 0) & (voxel_x < range_shape[1]) & (voxel_y >= 0) & (voxel_y < range_shape[0])
        voxel_x[~mask] = 0
        voxel_y[~mask] = 0

        self.voxel_x = torch.from_numpy(voxel_x).to(torch.int32)
        self.voxel_y = torch.from_numpy(voxel_y).to(torch.int32)
        self.mask = torch.from_numpy(mask).to(torch.bool)
        self.grid_shape = grid_shape
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.fuse_conv = self.make_res_layer(32, 32, 1, 1, 1, conv_cfg, norm_cfg, act_cfg)

    def make_conv_layer(self, in_channels: int, out_channels: int) -> None:  # two conv blocks in beginning
        return nn.Sequential(
            build_conv_layer(self.conv_cfg, in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(self.conv_cfg, out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg),
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

    def forward(self, bev_fea: Tensor, range_fea: Tensor):
        self.voxel_x = self.voxel_x.to(range_fea.device)
        self.voxel_y = self.voxel_y.to(range_fea.device)
        self.mask = self.mask.to(range_fea.device)
        range_fea_3d = range_fea[:, :, self.voxel_y, self.voxel_x]
        range_fea_3d[:, :, ~self.mask] = 0
        range_fea_3d = range_fea_3d.permute(0, 1, 4, 2, 3).contiguous()

        weight = self.fuse_conv(bev_fea)
        bev_fea_3d = bev_fea[:, None, :, :, :]
        weight = weight[:, None, :, :, :]
        fuse_fea = weight * range_fea_3d + bev_fea_3d
        # torch.cuda.empty_cache()
        return fuse_fea
