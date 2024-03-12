# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SemkittiRangeView(BaseTransform):
    """Convert Semantickitti point cloud dataset to range image."""

    def __init__(
        self,
        H: int = 64,
        W: int = 2048,
        fov_up: float = 3.0,
        fov_down: float = -25.0,
        means: Sequence[float] = (11.71279, -0.1023471, 0.4952, -1.0545, 0.2877),
        stds: Sequence[float] = (10.24, 12.295865, 9.4287, 0.8643, 0.1450),
        ignore_index: int = 19,
    ) -> None:
        self.H = H
        self.W = W
        self.fov_up = fov_up / 180.0 * np.pi
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov = abs(self.fov_down) + abs(self.fov_up)
        self.means = np.array(means, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)
        self.ignore_index = ignore_index

    def transform(self, results: dict) -> dict:
        points_numpy = results["points"].numpy()

        proj_image = np.full((self.H, self.W, 5), -1, dtype=np.float32)
        proj_idx = np.full((self.H, self.W), -1, dtype=np.int64)

        # get depth of all points
        depth = np.linalg.norm(points_numpy[:, :3], 2, axis=1)

        # get angles of all points
        yaw = -np.arctan2(points_numpy[:, 1], points_numpy[:, 0])
        pitch = np.arcsin(points_numpy[:, 2] / depth)

        # get projection in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov

        # scale to image size using angular resolution
        proj_x *= self.W
        proj_y *= self.H

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int64)

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int64)

        results["proj_x"] = proj_x
        results["proj_y"] = proj_y
        results["unproj_range"] = depth

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        proj_idx[proj_y[order], proj_x[order]] = indices[order]
        proj_image[proj_y[order], proj_x[order], 0] = depth[order]
        proj_image[proj_y[order], proj_x[order], 1:] = points_numpy[order]
        proj_mask = (proj_idx > 0).astype(np.int32)
        results["proj_range"] = proj_image[..., 0]

        proj_image = (proj_image - self.means[None, None, :]) / self.stds[None, None, :]
        proj_image = proj_image * proj_mask[..., None].astype(np.float32)
        results["img"] = proj_image

        if "pts_semantic_mask" in results:
            proj_sem_label = np.full((self.H, self.W), self.ignore_index, dtype=np.int64)
            proj_sem_label[proj_y[order], proj_x[order]] = results["pts_semantic_mask"][order]
            results["gt_semantic_seg"] = proj_sem_label

            # label_show = np.full((self.H, self.W, 3), self.ignore_index, dtype=np.uint8)
            # for i in range(20):
            #     label_show[proj_sem_label == i] = palette[i]

        return results


palette = list(
    [
        [100, 150, 245],
        [100, 230, 245],
        [30, 60, 150],
        [80, 30, 180],
        [100, 80, 250],
        [155, 30, 30],
        [255, 40, 200],
        [150, 30, 90],
        [255, 0, 255],
        [255, 150, 255],
        [75, 0, 75],
        [175, 0, 75],
        [255, 200, 0],
        [255, 120, 50],
        [0, 175, 0],
        [135, 60, 0],
        [150, 240, 80],
        [255, 240, 150],
        [255, 0, 0],
        [0, 0, 0],
    ]
)


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
