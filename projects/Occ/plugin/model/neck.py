import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence, Optional, List
from torch import Tensor
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmdet3d.utils import ConfigType, OptConfigType
from mmcv.cnn import ConvModule


@MODELS.register_module()
class FusionNet(BaseModule):
    def __init__(
        self,
        indices: int = 0,
        voxel_size: List = [0.2, 0.2, 0.2],
        pc_range: List = [0, -25.6, -2, 51.2, 25.6, 4.4],
        range_shape: List = [64, 512],
        range_fov: List = [-25.0, 3.0],
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN"),
        act_cfg: ConfigType = dict(type="LeakyReLU"),
    ):
        super().__init__()

        # First convolution
        # weight conv
        self.voxel_size = torch.tensor(voxel_size)
        self.pc_range = torch.tensor(pc_range)
        self.range_shape = torch.tensor(range_shape)
        self.range_fov = torch.tensor(range_fov)
        self.indices = indices

    def transform_3d2d(self, points: Tensor):
        H = self.range_shape[0].to(points.device)
        W = self.range_shape[1].to(points.device)
        fov_down = self.range_fov[0].to(points.device)
        fov_up = self.range_fov[1].to(points.device)

        fov_down_pi = fov_down / 180.0 * np.pi
        fov_up_pi = fov_up / 180.0 * np.pi
        fov_pi = abs(fov_down_pi) + abs(fov_up_pi)
        zero = torch.tensor(0, device=points.device)

        # get depth of all points
        depth = torch.norm(points[:, :3], 2, dim=1)

        # get angles of all points
        yaw = -torch.arctan2(points[:, 1], points[:, 0])
        pitch = torch.arcsin(points[:, 2] / (depth + 1e-6))

        # get projection in image coords
        proj_x = 0.5 * (yaw / torch.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(fov_down_pi)) / fov_pi

        # scale to image size using angular resolution
        proj_x *= W
        proj_y *= H

        # round and clamp for use as index
        proj_x = torch.floor(proj_x)
        proj_x = torch.minimum(W - 1, proj_x)
        proj_x = torch.maximum(zero, proj_x).to(torch.int64)

        proj_y = torch.floor(proj_y)
        proj_y = torch.minimum(H - 1, proj_y)
        proj_y = torch.maximum(zero, proj_y).to(torch.int64)

        return torch.stack([proj_y, proj_x], dim=1)

    def forward(self, geo_fea: Tensor, geo_pred: Tensor, range_fea: Tensor):
        voxel_size = self.voxel_size.to(geo_fea.device)
        pc_lowest = self.pc_range[:3].to(geo_fea.device)

        indices_grid = torch.nonzero(geo_pred)
        indices_3d = indices_grid[:, 1:] * voxel_size + pc_lowest

        indices_2d = self.transform_3d2d(indices_3d)
        sem_fea = range_fea[self.indices][indices_grid[:, 0], :, indices_2d[:, 0], indices_2d[:, 1]]

        # geo_fea_smaple = geo_fea.permute(0, 2, 3, 1).contiguous()[indices_grid[:, 0], indices_grid[:, 1], indices_grid[:, 2], indices_grid[:, 3]]
        # weight = self.weight_conv(geo_fea_smaple)
        # sem_fea = weight * sem_fea + geo_fea_smaple

        indices_grid_copy = indices_grid.clone()
        indices_grid[:, 0:3] = indices_grid_copy[:, 1:4]
        indices_grid[:, 3] = indices_grid_copy[:, 0]

        return sem_fea, indices_grid.to(torch.int)
