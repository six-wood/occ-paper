# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms import LoadAnnotations3D, LoadPointsFromFile
from mmdet3d.structures.points import BasePoints, get_points_type


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


x, y, z = np.meshgrid(
    np.arange(256),
    np.arange(256),
    np.arange(32),
    indexing="ij",
)
x = x.ravel()
y = y.ravel()
z = z.ravel()


def occupied_voxels(occupancy_grid, grid_shape, voxel_size, pc_lower_bound):
    """Saves only occupied voxels to a text file."""
    # Reshape the grid
    reshaped_grid = occupancy_grid.reshape(grid_shape)

    # Generate grid coordinates

    # Flatten and filter out unoccupied voxels
    coordinates = np.vstack((x, y, z, reshaped_grid.ravel())).T
    occupied_coordinates = coordinates[(coordinates[:, 3] > 0) & (coordinates[:, 3] < 255)].astype(np.float32)
    # occupied_coordinates = coordinates[(coordinates[:, 3] > 0)]

    # Save to text file
    occupied_coordinates[:, :3] = occupied_coordinates[:, :3] * voxel_size + pc_lower_bound

    return occupied_coordinates


@TRANSFORMS.register_module()
class LoadAnnotationsOcc(LoadAnnotations3D):
    def __init__(
        self,
        grid_size=[256, 256, 32],
        voxel_size=[0.2, 0.2, 0.2],
        pc_lower_bound=[-51.2, -25.6, -2.0],
    ) -> None:
        self.grid_size = np.array(grid_size)
        self.voxel_size = np.array(voxel_size)
        self.pc_lower_bound = np.array(pc_lower_bound)

    def transform(self, results: dict) -> dict:
        target = np.load(results["voxel_label_path"]["1_1"])
        target = target.reshape(-1).reshape(self.grid_size).astype(np.float32)
        results["voxel_label"] = target
        occ_points = occupied_voxels(target, self.grid_size, self.voxel_size, self.pc_lower_bound)
        results["pts_semantic_mask"] = occ_points[:, 3].astype(np.int64)

        # 'eval_ann_info' will be passed to evaluator
        if "eval_ann_info" in results:
            results["eval_ann_info"]["pts_semantic_mask"] = occ_points[:, 3].astype(np.int64)
        return results


@TRANSFORMS.register_module()
class LoadPointsFromFileOcc(LoadPointsFromFile):
    def __init__(
        self,
        grid_size=[256, 256, 32],
        voxel_size=[0.2, 0.2, 0.2],
        pc_lower_bound=[-51.2, -25.6, -2.0],
    ) -> None:
        self.grid_size = np.array(grid_size)
        self.voxel_size = np.array(voxel_size)
        self.pc_lower_bound = np.array(pc_lower_bound)

    def transform(self, results: dict) -> dict:
        target = np.load(results["voxel_label_path"]["1_1"])
        target = target.reshape(-1).reshape(self.grid_size).astype(np.float32)
        results["voxel_label"] = target
        occ_points = occupied_voxels(target, self.grid_size, self.voxel_size, self.pc_lower_bound)
        points = occ_points[:, :3]

        points_class = get_points_type("LIDAR")
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        results["points"] = points
        return results
