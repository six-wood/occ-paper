# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np

from typing import List, Sequence
from mmcv.transforms.base import BaseTransform
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadVoxelLabelFromFile(BaseTransform):
    def __init__(
        self,
        scale: str = "1_1",
        ignore_index: int = 255,
        grid_size: List = [256, 256, 32],
    ) -> None:
        self.scale = scale
        self.ignore_index = ignore_index
        self.grid_size = np.array(grid_size)

    def transform(self, results: dict) -> dict:
        target = np.load(results["voxel_label_path"][self.scale])
        target = target.reshape(-1).reshape(self.grid_size).astype(np.float32)
        results["voxel_label"] = target
        return results
