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


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, num_pos_feats):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Linear(input_channel, num_pos_feats),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_pos_feats, num_pos_feats),
        )

    def forward(self, xyz):
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


@MODELS.register_module()
class FusionNet(BaseModule):
    def __init__(
        self,
        indices: int = 0,
        pose_embending_dim: int = 128,
        voxel_size: List = [0.2, 0.2, 0.2],
        pc_range: List = [0, -25.6, -2, 51.2, 25.6, 4.4],
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN"),
        act_cfg: ConfigType = dict(type="LeakyReLU"),
    ):
        super().__init__()

        # First convolution
        # weight conv
        self.voxel_size = torch.tensor(voxel_size)
        self.pc_range = torch.tensor(pc_range)
        self.indices = indices

        self.pose_embending = PositionEmbeddingLearned(3, pose_embending_dim)

    def forward(self, geo_fea: Tensor, geo_pred: Tensor):
        voxel_size = self.voxel_size.to(geo_fea.device)
        pc_lowest = self.pc_range[:3].to(geo_fea.device)

        indices_grid = torch.nonzero(geo_pred)
        indices_3d = indices_grid[:, 1:] * voxel_size + pc_lowest
        pos_query = self.pose_embending(indices_3d)

        sem_fea = pos_query

        # geo_fea_smaple = geo_fea.permute(0, 2, 3, 1).contiguous()[indices_grid[:, 0], indices_grid[:, 1], indices_grid[:, 2], indices_grid[:, 3]]
        # weight = self.weight_conv(geo_fea_smaple)
        # sem_fea = weight * sem_fea + geo_fea_smaple

        # swap x and z
        indices_grid_copy = indices_grid.clone()
        indices_grid[:, 1] = indices_grid_copy[:, 3]
        indices_grid[:, 3] = indices_grid_copy[:, 1]

        return sem_fea, indices_grid.to(torch.int)
