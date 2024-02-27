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
from .moudle import ASPPBlock
from .data_preprocessor import SemkittiRangeView
from .moudle import compute_visibility_mask


@MODELS.register_module()
class FusionNet(BaseModule):
    def __init__(
        self,
        voxel_size: Sequence[float] = [0.2, 0.2, 0.2],
        range_shape: Sequence[int] = [64, 1024],
        pc_range: Sequence[float] = [0, -25.6, -2.0, 51.2, 25.6, 4.4],
        range_fov: Sequence[float] = [-25.0, -90.0, 3.0, 90.0],
        fusion_layer: Optional[dict] = None,
        bev_inplanes: int = 1,
        bev_planes: int = 8,
        bev_outplanes: int = 2,
        dilations=(1, 2, 3),
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN"),
        act_cfg: ConfigType = dict(type="LeakyReLU"),
    ):
        super().__init__()
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
        self.bev_conv0 = ConvModule(bev_inplanes, bev_planes, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.aspp3d = ASPPBlock(
            in_channels=bev_planes,
            channels=bev_planes,
            dilations=dilations,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.bev_class = ConvModule(bev_planes, bev_outplanes, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)

        if fusion_layer is not None:
            self.fusion_layer = MODELS.build(fusion_layer)

    def forward(self, bev_fea: Tensor, range_fea: Tensor):
        B, Z, H, W = bev_fea.shape
        device = bev_fea.device
        self.visibility_mask = self.visibility_mask.to(device)
        self.voxel_size = self.voxel_size.to(device)
        self.offset = self.offset.to(device)
        x = bev_fea[:, None, :, :, :]
        x = self.bev_conv0(x)
        x = self.aspp3d(x)
        x = self.bev_class(x)
        # geometric feature
        geo_fea = torch.permute(x, (0, 1, 3, 4, 2)).contiguous()  # B, C, Z, H, W -> B, C, H, W, Z
        # semantic feature
        sc_prob = F.softmax(geo_fea, dim=1)[:, 1]
        sc_prob = torch.where(self.visibility_mask[None, :], sc_prob, 0)
        sc_prob = sc_prob.view(B, -1)  # B, H, W, Z -> B, H*W*Z
        _, sc_query = torch.topk(sc_prob, dim=1, k=H * W * 4, largest=True)  # rank in the range of 0 to H*W*Z
        sc_query_grid_coor = torch.stack([sc_query // (W * Z), (sc_query % (W * Z)) // Z, (sc_query % (W * Z)) % Z], dim=2) # B, N, 3
        sc_query_points = sc_query_grid_coor * self.voxel_size + self.voxel_size / 2 + self.offset
        sc_query_range = self.transform_range_coord(sc_query_points).unsqueeze(1)
        sem_fea = F.grid_sample(range_fea, sc_query_range, align_corners=False).squeeze(2)  # B, C, N
        # # test
        # r = torch.zeros_like(x).to(torch.float32)
        # p = F.softmax(x, dim=1)
        # r.scatter_(2, sc_query[:, None, :].expand(-1, 2, -1), values[:, None, :].expand(-1, 2, -1))
        return geo_fea, sem_fea, sc_query_grid_coor