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
class BevNet(BaseModule):
    def __init__(
        self,
        bev_shape: Sequence[int] = [256, 256, 32],
        range_shape: Sequence[int] = [64, 1024],
        pc_range: Sequence[float] = [0, -25.6, -2.0, 51.2, 25.6, 4.4],
        range_fov: Sequence[float] = [-90, -25, 90, 2],
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN"),
        act_cfg: ConfigType = dict(type="LeakyReLU"),
    ):
        super().__init__()
        voxel_grid_shape = np.array(bev_shape)
        voxel_size = (np.array(pc_range[3:]) - np.array(pc_range[:3])) / voxel_grid_shape
        xx, yy, zz = np.meshgrid(
            np.linspace(pc_range[0] + voxel_size[0] / 2, pc_range[3] - voxel_size[0] / 2, voxel_grid_shape[0]),
            np.linspace(pc_range[1] + voxel_size[1] / 2, pc_range[4] - voxel_size[1] / 2, voxel_grid_shape[1]),
            np.linspace(pc_range[2] + voxel_size[2] / 2, pc_range[5] - voxel_size[2] / 2, voxel_grid_shape[2]),
            indexing="ij",
        )
        r_voxel = np.sqrt(xx**2 + yy**2 + zz**2)
        yaw_voxel = np.arctan2(yy, xx)
        pit_voxel = np.arcsin(zz / r_voxel)

        fov_left = range_fov[0] * np.pi / 180
        fov_down = range_fov[1] * np.pi / 180
        fov_x = (range_fov[2] - range_fov[0]) * np.pi / 180
        fov_y = (range_fov[3] - range_fov[1]) * np.pi / 180

        voxel_x = ((yaw_voxel + abs(fov_left)) / fov_x) * range_shape[1]
        voxel_y = ((pit_voxel + abs(fov_down)) / fov_y) * range_shape[0]

        mask = (voxel_x >= 0) & (voxel_x < range_shape[1]) & (voxel_y >= 0) & (voxel_y < range_shape[0])
        voxel_x[~mask] = -1
        voxel_y[~mask] = -1
        self.voxel_x = voxel_x.astype(np.int32)
        self.voxel_y = voxel_y.astype(np.int32)
