import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence, Optional
from torch import Tensor
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmdet3d.utils import ConfigType, OptConfigType
from mmcv.cnn import ConvModule
from .data_preprocessor import SemkittiRangeView
from .moudle import compute_visibility_mask


@MODELS.register_module()
class FusionNet(BaseModule):
    def __init__(
        self,
        top_k_scatter: int = 4,
        voxel_size: Sequence[float] = [0.2, 0.2, 0.2],
        range_shape: Sequence[int] = [64, 1024],
        pc_range: Sequence[float] = [0, -25.6, -2.0, 51.2, 25.6, 4.4],
        range_fov: Sequence[float] = [-25.0, -90.0, 3.0, 90.0],
        fusion_layer: Optional[dict] = None,
        bev_inplanes: int = 1,
        bev_outplanes: int = 2,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN"),
        act_cfg: ConfigType = dict(type="LeakyReLU"),
    ):
        super().__init__()
        self.top_k_scatter = top_k_scatter
        self.voxel_size = torch.from_numpy(np.array(voxel_size)).to(torch.float32)
        self.offset = torch.from_numpy(np.array(pc_range[:3])).to(torch.float32)
        visibility_mask = compute_visibility_mask(
            center=[0, 0, 0],
            pc_range=pc_range,
            voxel_size=voxel_size,
            fov=[range_fov[0], range_fov[2]],
        )
        self.visibility_mask = torch.from_numpy(visibility_mask).to(torch.bool)
        self.transform_range_coord = SemkittiRangeView(
            H=range_shape[0],
            W=range_shape[1],
            fov_up=range_fov[2],
            fov_down=range_fov[0],
            fov_left=range_fov[1],
            fov_right=range_fov[3],
        ).get_range_norm_coord

        # First convolution
        self.sc_class = ConvModule(bev_inplanes, bev_outplanes, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)
        # weight conv
        self.weight_conv = ConvModule(bev_inplanes, 1, 1, conv_cfg=dict(type=nn.Conv1d), norm_cfg=norm_cfg, act_cfg=act_cfg)

        if fusion_layer is not None:
            self.fusion_layer = MODELS.build(fusion_layer)

    def forward(self, bev_fea: Tensor, range_fea: Tensor):
        # init
        B, Z, H, W = bev_fea.shape
        device = bev_fea.device
        self.visibility_mask = self.visibility_mask.to(device)
        self.voxel_size = self.voxel_size.to(device)
        self.offset = self.offset.to(device)

        # geometric feature
        x = bev_fea[:, None, :, :, :]
        x = self.sc_class(x)
        geo_fea = torch.permute(x, (0, 1, 3, 4, 2)).contiguous()  # B, C, Z, H, W -> B, C, H, W, Z

        # semantic feature
        sc_prob = F.softmax(geo_fea, dim=1)[:, 1]
        sc_prob = torch.where(self.visibility_mask[None, :], sc_prob, 0)
        sc_prob = sc_prob.view(B, -1)  # B, H, W, Z -> B, H*W*Z
        _, sc_query = torch.topk(sc_prob, dim=1, k=H * W * self.top_k_scatter, largest=True)  # rank in the range of 0 to H*W*Z
        sc_query_grid_coor = torch.stack([sc_query // (W * Z), (sc_query % (W * Z)) // Z, (sc_query % (W * Z)) % Z], dim=2)  # B, N, 3
        sc_query_points = sc_query_grid_coor * self.voxel_size + self.voxel_size / 2 + self.offset

        sc_query_range = self.transform_range_coord(sc_query_points).unsqueeze(1)
        sem_fea = F.grid_sample(range_fea, sc_query_range, align_corners=False).squeeze(2)  # B, C, N

        # geo weight(for occlusion
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, sc_query_grid_coor.shape[1]).unsqueeze(2)
        sc_query_grid_coor = torch.cat([batch_indices, sc_query_grid_coor], dim=2).to(torch.int32)
        bev_fea = bev_fea.permute(0, 2, 3, 1).contiguous()
        geo_fea_sample = bev_fea[
            sc_query_grid_coor[:, :, 0],
            sc_query_grid_coor[:, :, 1],
            sc_query_grid_coor[:, :, 2],
            sc_query_grid_coor[:, :, 3],
        ].unsqueeze(1)
        weight = self.weight_conv(geo_fea_sample)
        sem_fea = weight * sem_fea + geo_fea_sample

        return geo_fea, sem_fea, sc_query_grid_coor
