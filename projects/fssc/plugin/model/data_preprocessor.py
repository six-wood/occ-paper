# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor


@MODELS.register_module()
class OccDDataPreprocessor(Det3DDataPreprocessor):
    @torch.no_grad()
    def voxelize(self, points: List[Tensor], data_samples: SampleList) -> Dict[str, Tensor]:
        """Apply voxelization to point cloud.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (list[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            Dict[str, Tensor]: Voxelization information.

            - voxels (Tensor): Features of voxels, shape is MxNxC for hard
              voxelization, NxC for dynamic voxelization.
            - coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
              where 1 represents the batch index.
            - num_points (Tensor, optional): Number of points in each voxel.
            - voxel_centers (Tensor, optional): Centers of voxels.
        """

        voxel_dict = dict()

        coors = []
        # dynamic voxelization only provide a coors mapping
        for i, (res, data_sample) in enumerate(zip(points, data_samples)):
            res_coors = self.voxel_layer(res)
            res_coors = F.pad(res_coors, (1, 0), mode="constant", value=i)
            self.get_voxel_seg(res_coors, data_sample)
            coors.append(res_coors)
        voxels = torch.cat(points, dim=0)
        coors = torch.cat(coors, dim=0)

        voxel_dict["voxels"] = voxels
        voxel_dict["coors"] = coors

        return voxel_dict
