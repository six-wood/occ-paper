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


@MODELS.register_module()
class FusionNet(BaseModule):
    def __init__(
        self,
        grid_shape: Sequence[int] = [256, 256, 32],
        range_shape: Sequence[int] = [64, 1024],
        pc_range: Sequence[float] = [0, -25.6, -2.0, 51.2, 25.6, 4.4],
        range_fov: Sequence[float] = [-25, -90, 3, 90],
        fusion_layer: Optional[dict] = None,
        bev_inplanes: int = 1,
        bev_planes: int = 8,
        bev_outplanes: int = 2,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN"),
        act_cfg: ConfigType = dict(type="LeakyReLU"),
    ):
        super().__init__()
        voxel_grid_shape = np.array(grid_shape)
        voxel_size = (np.array(pc_range[3:]) - np.array(pc_range[:3])) / voxel_grid_shape
        self.voxel_size = torch.from_numpy(voxel_size).to(torch.float32)
        xx, yy, zz = np.meshgrid(
            np.linspace(pc_range[0] + voxel_size[0] / 2, pc_range[3] - voxel_size[0] / 2, voxel_grid_shape[0]),
            np.linspace(pc_range[1] + voxel_size[1] / 2, pc_range[4] - voxel_size[1] / 2, voxel_grid_shape[1]),
            np.linspace(pc_range[2] + voxel_size[2] / 2, pc_range[5] - voxel_size[2] / 2, voxel_grid_shape[2]),
            indexing="ij",
        )
        r_voxel = np.sqrt(xx**2 + yy**2 + zz**2)
        yaw_voxel = -np.arctan2(yy, xx)
        pit_voxel = np.arcsin(zz / r_voxel)

        fov_down = range_fov[0] * np.pi / 180
        fov_up = range_fov[2] * np.pi / 180
        fov_left = range_fov[1] * np.pi / 180
        fov_right = range_fov[3] * np.pi / 180
        mask = (pit_voxel >= fov_down) & (pit_voxel < fov_up) & (yaw_voxel >= fov_left) & (yaw_voxel < fov_right)
        self.mask = torch.from_numpy(mask).to(torch.bool)
        self.transform_range_coord = SemkittiRangeView(
            H=range_shape[0],
            W=range_shape[1],
            fov_up=range_fov[2],
            fov_down=range_fov[0],
            fov_left=range_fov[1],
            fov_right=range_fov[3],
        ).get_range_view_coord

        # First convolution
        self.bev_conv0 = ConvModule(bev_inplanes, bev_planes, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.aspp3d = ASPPBlock(
            in_channels=bev_planes,
            channels=bev_planes,
            dilations=(1, 2, 3),
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
        self.mask = self.mask.to(device)
        self.voxel_size = self.voxel_size.to(device)
        x = bev_fea[:, None, :, :, :]
        x = self.bev_conv0(x)
        x = self.aspp3d(x)
        x = self.bev_class(x)
        # geometric feature
        geo_fea = torch.permute(x, (0, 1, 3, 4, 2)).contiguous()
        # semantic feature
        sc_prob = F.softmax(geo_fea, dim=1)[:, 1]
        sc_prob = torch.where(self.mask[None, :], sc_prob, 0)
        sc_prob = sc_prob.view(B, -1)  # B, C, Z, H, W -> B, C, H, W, Z -> B, C, H*W*Z
        values, sc_query = torch.topk(sc_prob, dim=1, k=H * W * 4, largest=True)  # rank in the range of 0 to H*W*Z
        sc_query_grid_coor = torch.stack([sc_query // (W * Z), (sc_query % (W * Z)) // Z, (sc_query % (W * Z)) % Z], dim=2)
        sc_query_points = sc_query_grid_coor * self.voxel_size + self.voxel_size / 2
        sc_query_range = self.transform_range_coord(sc_query_points)
        sem_fea = geo_fea.gather(2, sc_query[:, None, :].expand(-1, 2, -1))  # B, C, H*W*Z -> B, C, H*W*4

        # # test
        # r = torch.zeros_like(x).to(torch.float32)
        # p = F.softmax(x, dim=1)
        # r.scatter_(2, sc_query[:, None, :].expand(-1, 2, -1), values[:, None, :].expand(-1, 2, -1))

        return geo_fea, sem_fea, sc_query
