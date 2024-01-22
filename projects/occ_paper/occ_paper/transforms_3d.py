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


@TRANSFORMS.register_module()
class SemkittiRangeView(BaseTransform):
    """Convert Semantickitti point cloud dataset to range image."""

    def __init__(
        self,
        H: int = 64,
        W: int = 1024,
        fov_up: float = 3.0,
        fov_down: float = -25.0,
        fov_left: float = -90.0,
        fov_right: float = 90.0,
        means: Sequence[float] = (11.71279, -0.1023471, 0.4952, -1.0545, 0.2877),
        stds: Sequence[float] = (10.24, 12.295865, 9.4287, 0.8643, 0.1450),
        ignore_index: int = 255,
    ) -> None:
        self.H = H
        self.W = W
        self.fov_up = fov_up / 180.0 * np.pi
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov_left = fov_left / 180.0 * np.pi
        self.fov_right = fov_right / 180.0 * np.pi
        self.fov_y = abs(self.fov_down) + abs(self.fov_up)
        self.fov_x = abs(self.fov_right) + abs(self.fov_left)
        self.means = np.array(means, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)
        self.ignore_index = ignore_index

    def transform(self, results: dict) -> dict:
        proj_image = np.full((self.H, self.W, 5), -1, dtype=np.float32)
        proj_idx = np.full((self.H, self.W), -1, dtype=np.int64)

        points_numpy = results["points"].numpy()

        # get depth of all points
        depth = np.linalg.norm(points_numpy[:, :3], 2, axis=1)

        # get angles of all points
        yaw = -np.arctan2(points_numpy[:, 1], points_numpy[:, 0])
        pitch = np.arcsin(points_numpy[:, 2] / depth)

        # filter based on angles
        fov_filter = np.logical_and(
            np.logical_and((pitch < self.fov_up), (pitch > self.fov_down)),
            np.logical_and((yaw < self.fov_right), (yaw > self.fov_left)),
        )
        depth = depth[fov_filter]
        yaw = yaw[fov_filter]
        pitch = pitch[fov_filter]
        points_numpy = points_numpy[fov_filter]

        # get projection in image coords
        proj_x = (yaw + abs(self.fov_left)) / self.fov_x
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov_y

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
        show_depth = proj_image[..., 0].copy()
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
        return results


@TRANSFORMS.register_module()
class ApplayVisMask(BaseTransform):
    """Apply visibility mask to range image."""

    def compute_visibility_mask(
        self,
        center: np.ndarray,
        pc_range: np.ndarray,
        voxel_size: np.ndarray,
        fov: np.ndarray,
    ) -> np.ndarray:
        # 计算网格大小
        grid_size = np.round((pc_range[3:] - pc_range[:3]) / voxel_size).astype(np.int32)

        # 确定每个轴的范围
        x_range = np.linspace(pc_range[0], pc_range[3], grid_size[0])
        y_range = np.linspace(pc_range[1], pc_range[4], grid_size[1])
        z_range = np.linspace(pc_range[2], pc_range[5], grid_size[2])

        # 生成三维网格
        xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing="ij")

        # 调整网格以反映中心点的偏移
        xx -= center[0]
        yy -= center[1]
        zz -= center[2]

        # 计算每个点的俯仰角
        pitch_angles = np.arctan2(zz, np.sqrt(xx**2 + yy**2))

        # 转换为度
        pitch_angles_degrees = np.degrees(pitch_angles)

        # 确定每个体素是否在视场范围内
        visibility_mask = (pitch_angles_degrees >= fov[0]) & (pitch_angles_degrees <= fov[1])

        return visibility_mask

    def __init__(
        self,
        center: np.array = np.array([0.0, 0.0, 0.0]),
        pc_range: np.array = np.array([0.0, -25.6, -2.0, 51.2, 25.6, 4.4]),
        voxel_size: np.array = np.array([0.2, 0.2, 0.2]),
        fov: np.array = np.array([-25, 3]),
        **kwargs,
    ) -> None:
        self.visibility_mask = self.compute_visibility_mask(center, pc_range, voxel_size, fov)

    def transform(self, results: dict) -> dict:
        if "voxel_label" in results:
            results["voxel_label"] = np.where(self.visibility_mask, results["voxel_label"], 255)
        return results
