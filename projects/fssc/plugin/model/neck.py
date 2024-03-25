import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch import Tensor
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmdet3d.utils import ConfigType, OptConfigType
from spconv.pytorch import SparseConvTensor
from spconv.pytorch.functional import sparse_add_hash_based


class ScEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, channels):
        super().__init__()
        self.position_embedding_head = nn.Sequential()
        for i, channel in enumerate(channels):
            self.position_embedding_head.add_module(f"linear{i}", nn.Linear(input_channel, channel))
            self.position_embedding_head.add_module(f"bn{i}", nn.BatchNorm1d(channel))
            self.position_embedding_head.add_module(f"relu{i}", nn.ReLU())
            input_channel = channel
        self.position_embedding_head.add_module("linear", nn.Linear(channels[-1], channels[-1]))

    def forward(self, sc_points):
        position_embedding = self.position_embedding_head(sc_points)
        return position_embedding


@MODELS.register_module()
class SampleNet(BaseModule):
    def __init__(
        self,
        top_k_scatter: int = 8,
        sc_embedding_dim: List[int] = [16, 32],
        voxel_size: List = [0.2, 0.2, 0.2],
        pc_range: List = [0, -25.6, -2, 51.2, 25.6, 4.4],
    ):
        super().__init__()

        self.top_k_scatter = top_k_scatter
        # weight conv
        self.voxel_size = torch.tensor(voxel_size)
        self.offset = torch.tensor(pc_range[:3])
        self.sc_embedding = ScEmbeddingLearned(4, sc_embedding_dim)

    def forward(self, geo_probs: Tensor, bev_fea: Tensor, pts_feats: Tensor, pts_coors: Tensor) -> SparseConvTensor:
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

        spatial_shape = sc_coors.max(0)[0][1:] + 1
        batch_size = int(sc_coors[-1, 0]) + 1
        sc_embeding = self.sc_embedding(sc_points)
        bev_fea = SparseConvTensor(sc_embeding, sc_coors, spatial_shape, batch_size)
        pts_fea = SparseConvTensor(pts_feats, pts_coors, spatial_shape, batch_size)

        x = sparse_add_hash_based(bev_fea, pts_fea)

        return x
