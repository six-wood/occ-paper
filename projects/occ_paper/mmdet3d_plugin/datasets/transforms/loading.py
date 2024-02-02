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
        task: str = "sc",
        scale: str = "1_1",
        ignore_index: int = 255,
        grid_size: List = [256, 256, 32],
    ) -> None:
        self.task = task
        self.scale = scale
        self.ignore_index = ignore_index
        self.grid_size = np.array(grid_size)

    def transform(self, results: dict) -> dict:
        target = np.load(results["voxel_label_path"][self.scale])
        target = target.reshape(-1).reshape(self.grid_size).astype(np.float32)

        assert self.task in ["sc", "ssc"]
        if self.task == "sc":
            ones = np.ones_like(target)
            target = np.where(np.logical_or(target == self.ignore_index, target == 0), target, ones)

        results["voxel_label"] = target
        return results
