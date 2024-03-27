# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from functools import partial
from typing import List

import torch
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import Tensor, nn


from mmdet3d.models.layers.sparse_block import SparseBasicBlock, SparseBottleneck, make_sparse_convmodule
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE

from mmdet3d.utils import OptMultiConfig, ConfigType

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor


@MODELS.register_module()
class PtsNet(BaseModule):
    r"""MinkUNet backbone with TorchSparse backend.

    Refer to `implementation code <https://github.com/mit-han-lab/spvnas>`_.

    Args:
        in_channels (int): Number of input voxel feature channels.
            Defaults to 4.
        base_channels (int): The input channels for first encoder layer.
            Defaults to 32.
        num_stages (int): Number of stages in encoder and decoder.
            Defaults to 4.
        encoder_channels (List[int]): Convolutional channels of each encode
            layer. Defaults to [32, 64, 128, 256].
        encoder_blocks (List[int]): Number of blocks in each encode layer.
        decoder_channels (List[int]): Convolutional channels of each decode
            layer. Defaults to [256, 128, 96, 96].
        decoder_blocks (List[int]): Number of blocks in each decode layer.
        block_type (str): Type of block in encoder and decoder.
        sparseconv_backend (str): Sparse convolutional backend.
        init_cfg (dict or :obj:`ConfigDict` or List[dict or :obj:`ConfigDict`]
            , optional): Initialization config dict.
    """

    def __init__(
        self,
        pts_voxel_encoder: ConfigType = None,
        in_channels: int = 16,
        base_channels: int = 16,
        num_stages: int = 1,
        encoder_channels: List[int] = [16],
        encoder_blocks: List[int] = [1],
        block_type: str = "basic",
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg)
        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
        assert num_stages == len(encoder_channels)
        self.num_stages = num_stages

        input_conv = partial(make_sparse_convmodule, conv_type="SubMConv3d")
        residual_block = SparseBasicBlock if block_type == "basic" else SparseBottleneck
        residual_branch = partial(make_sparse_convmodule, conv_type="SubMConv3d", order=("conv", "norm"))

        self.conv_input = nn.Sequential(
            input_conv(in_channels, base_channels, kernel_size=3, padding=1, indice_key="subm0"),
            input_conv(base_channels, base_channels, kernel_size=3, padding=1, indice_key="subm0"),
        )

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        encoder_channels.insert(0, base_channels)

        for i in range(num_stages):
            encoder_layer = []
            for j in range(encoder_blocks[i]):
                if j == 0 and encoder_channels[i] != encoder_channels[i + 1]:
                    encoder_layer.append(
                        residual_block(
                            encoder_channels[i],
                            encoder_channels[i + 1],
                            downsample=residual_branch(encoder_channels[i], encoder_channels[i + 1], kernel_size=1)
                            if residual_branch is not None
                            else None,
                            indice_key=f"subm{i+1}",
                        )
                    )
                else:
                    encoder_layer.append(residual_block(encoder_channels[i + 1], encoder_channels[i + 1], indice_key=f"subm{i+1}"))
            self.encoder.append(nn.Sequential(*encoder_layer))

    def forward(self, feas: Tensor, coors: Tensor) -> Tensor:
        """Forward function.

        Args:
            voxel_features (Tensor): Voxel features in shape (N, C).
            coors (Tensor): Coordinates in shape (N, 4),
                the columns in the order of (x_idx, y_idx, z_idx, batch_idx).

        Returns:
            Tensor: Backbone features.
        """

        voxel_features, coors = self.pts_voxel_encoder(feas, coors)

        spatial_shape = coors.max(0)[0][1:] + 1
        batch_size = int(coors[-1, 0]) + 1
        x = SparseConvTensor(voxel_features, coors, spatial_shape, batch_size)

        x = self.conv_input(x)
        for encoder_layer in self.encoder:
            x = encoder_layer(x)

        return x.features, coors
