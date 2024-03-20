import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch import Tensor
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmdet3d.utils import ConfigType, OptConfigType


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
class DownsampleNet(BaseModule):
    def __init__(
        self,
        top_k_scatter: int = 8,
        voxel_size: List = [0.2, 0.2, 0.2],
        pc_range: List = [0, -25.6, -2, 51.2, 25.6, 4.4],
    ):
        super().__init__()

        self.top_k_scatter = top_k_scatter
        # weight conv
        self.voxel_size = torch.tensor(voxel_size)
        self.offset = torch.tensor(pc_range[:3])

    def forward(self, geo_probs: Tensor, bev_fea: Tensor):
        B, _, H, W, D = geo_probs.shape
        K = H * W * self.top_k_scatter
        device = geo_probs.device
        self.voxel_size = self.voxel_size.to(device)
        self.offset = self.offset.to(device)

        # semantic feature
        sc_prob = geo_probs[:, 1].view(B, -1)  # B, H, W, D -> B, H*W*D
        _, sc_topk = torch.topk(sc_prob, dim=1, k=K, largest=True)  # rank in the range of 0 to H*W*Z

        bev_fea = bev_fea.permute(0, 2, 3, 1).contiguous().view(B, -1)
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, K)
        geo_fea = bev_fea[batch_indices, sc_topk]  # B, N, C

        sc_coors = torch.stack([sc_topk // (W * D), (sc_topk % (W * D)) // D, (sc_topk % (W * D)) % D], dim=2)  # B, N, 3
        sc_points = sc_coors * self.voxel_size + self.voxel_size / 2 + self.offset
        sc_points = torch.cat([sc_points, geo_fea.unsqueeze(2)], dim=2).view(-1, 4)

        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, sc_coors.shape[1]).unsqueeze(2)
        sc_coors = torch.cat([batch_indices, sc_coors], dim=2).to(torch.int32).view(-1, 4)

        # swap x, z
        sc_copy = sc_coors.clone()
        sc_coors[:, 3] = sc_copy[:, 1]
        sc_coors[:, 1] = sc_copy[:, 3]

        return sc_points, sc_coors
