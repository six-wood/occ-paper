# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.datasets.transforms.transforms_3d import RandomFlip3D
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomFlipVoxel(RandomFlip3D):
    """Flip the points randomly in voxel level."""

    def __init__(
        self,
        sync_2d: bool = True,
        flip_ratio_bev_horizontal: float = 0.0,
        flip_ratio_bev_vertical: float = 0.0,
        flip_box3d: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(sync_2d, flip_ratio_bev_horizontal, flip_ratio_bev_vertical, flip_box3d, **kwargs)

    def random_flip_data_3d(self, input_dict: dict, direction: str = "horizontal") -> None:
        """Flip 3D data randomly.

        `random_flip_data_3d` should take these situations into consideration:

        - 1. LIDAR-based 3d detection
        - 2. LIDAR-based 3d segmentation
        - 3. vision-only detection
        - 4. multi-modality 3d detection.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Defaults to 'horizontal'.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are
            updated in the result dict.
        """
        assert direction in ["horizontal", "vertical"]

        if "centers_2d" in input_dict:
            assert self.sync_2d is True and direction == "horizontal", "Only support sync_2d=True and horizontal flip with images"
            w = input_dict["img_shape"][1]
            input_dict["centers_2d"][..., 0] = w - input_dict["centers_2d"][..., 0]
            # need to modify the horizontal position of camera center
            # along u-axis in the image (flip like centers2d)
            # ['cam2img'][0][2] = c_u
            # see more details and examples at
            # https://github.com/open-mmlab/mmdetection3d/pull/744
            input_dict["cam2img"][0][2] = w - input_dict["cam2img"][0][2]
        if "voxel_label" in input_dict:
            input_dict["points"].flip(direction)
            if direction == "horizontal":
                input_dict["voxel_label"] = np.fliplr(input_dict["voxel_label"])
            elif direction == "vertical":
                input_dict["voxel_label"] = np.flipud(input_dict["voxel_label"])


def compute_visibility_mask(
    center: list = [0, 0, 0],
    pc_range: list = [0, -25.6, -2.0, 51.2, 25.6, 4.4],
    voxel_size: list = [0.2, 0.2, 0.2],
    fov: list = [-25.0, 3.0],
) -> np.ndarray:
    # 计算网格大小
    pc_range = np.array(pc_range)
    voxel_size = np.array(voxel_size)
    fov = np.array(fov)
    grid_size = np.round((pc_range[3:] - pc_range[:3]) / voxel_size).astype(np.int32)

    # 确定每个轴的范围
    x_range = np.linspace(pc_range[0] + voxel_size[0] / 2, pc_range[3] - voxel_size[0] / 2, grid_size[0])
    y_range = np.linspace(pc_range[1] + voxel_size[1] / 2, pc_range[4] - voxel_size[1] / 2, grid_size[1])
    z_range = np.linspace(pc_range[2] + voxel_size[2] / 2, pc_range[5] - voxel_size[2] / 2, grid_size[2])

    # 生成三维网格
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing="ij")

    # 调整网格以反映中心点的偏移
    xx -= center[0]
    yy -= center[1]
    zz -= center[2]

    # 计算每个点的俯仰角
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    pitch_angles = np.arcsin(zz / r)

    # 转换为度
    pitch_angles_degrees = np.degrees(pitch_angles)

    # 确定每个体素是否在视场范围内
    visibility_mask = (pitch_angles_degrees >= fov[0]) & (pitch_angles_degrees <= fov[1])

    return visibility_mask


@TRANSFORMS.register_module()
class ApplayVisMask(BaseTransform):
    """Apply visibility mask to range image."""

    def __init__(
        self,
        center: list = [0.0, 0.0, 0.0],
        pc_range: list = [0.0, -25.6, -2.0, 51.2, 25.6, 4.4],
        voxel_size: list = [0.2, 0.2, 0.2],
        fov: list = [-25, 3],
        **kwargs,
    ) -> None:
        center = np.array(center)
        pc_range = np.array(pc_range)
        voxel_size = np.array(voxel_size)
        fov = np.array(fov)
        self.visibility_mask = compute_visibility_mask(center, pc_range, voxel_size, fov)

    def transform(self, results: dict) -> dict:
        if "voxel_label" in results:
            results["voxel_label"] = np.where(self.visibility_mask, results["voxel_label"], 255)
        return results
