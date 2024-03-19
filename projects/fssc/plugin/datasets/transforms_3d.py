# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadVoxelLabelFromFile(BaseTransform):
    def __init__(
        self,
        grid_size=[256, 256, 32],
    ) -> None:
        self.grid_size = np.array(grid_size)

    def transform(self, results: dict) -> dict:
        target = np.load(results["voxel_label_path"]["1_1"])
        target = target.reshape(-1).reshape(self.grid_size).astype(np.float32)
        results["voxel_label"] = target
        return results


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
class ApplyVisMask(BaseTransform):
    """Apply visibility mask to range image."""

    def __init__(
        self,
        center: list = [0.0, 0.0, 0.0],
        pc_range: list = [0.0, -25.6, -2.0, 51.2, 25.6, 4.4],
        voxel_size: list = [0.2, 0.2, 0.2],
        fov: list = [-25.0, 3.0],
        **kwargs,
    ) -> None:
        center = np.array(center)
        pc_range = np.array(pc_range)
        voxel_size = np.array(voxel_size)
        fov = np.array(fov)
        self.visibility_mask = compute_visibility_mask(center, pc_range, voxel_size, fov)

    def transform(self, results: dict) -> dict:
        results["voxel_label"] = np.where(self.visibility_mask, results["voxel_label"], 255)
        return results
