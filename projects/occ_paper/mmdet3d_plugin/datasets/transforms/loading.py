# Copyright (c) OpenMMLab. All rights reserved.
import torch
import mmengine
import numpy as np

from typing import List, Optional, Union
from mmcv.transforms.base import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms import LoadPointsFromFile, LoadAnnotations3D
from mmdet3d.structures.points import get_points_type
from mmengine.fileio import get


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


@TRANSFORMS.register_module()
class LoadScPointsFromFile(LoadPointsFromFile):
    def __init__(
        self,
        coord_type: str,
        load_dim: int = 6,
        use_dim: Union[int, List[int]] = [0, 1, 2],
        shift_height: bool = False,
        use_color: bool = False,
        norm_intensity: bool = False,
        norm_elongation: bool = False,
        backend_args: Optional[dict] = None,
    ) -> None:
        super(LoadScPointsFromFile, self).__init__(
            coord_type,
            load_dim,
            use_dim,
            shift_height,
            use_color,
            norm_intensity,
            norm_elongation,
            backend_args,
        )

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_file_path = results["sc_points"]["sc_path"]
        points = self._load_points(pts_file_path)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        if self.norm_intensity:
            assert len(self.use_dim) >= 4, f"When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}"  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        if self.norm_elongation:
            assert len(self.use_dim) >= 5, f"When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}"  # noqa: E501
            points[:, 4] = np.tanh(points[:, 4])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate([points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results["points"] = points

        return results


@TRANSFORMS.register_module()
class LoadScAnnotations3D(LoadAnnotations3D):
    def __init__(
        self,
        with_bbox_3d: bool = True,
        with_label_3d: bool = True,
        with_attr_label: bool = False,
        with_mask_3d: bool = False,
        with_seg_3d: bool = False,
        with_bbox: bool = False,
        with_label: bool = False,
        with_mask: bool = False,
        with_seg: bool = False,
        with_bbox_depth: bool = False,
        with_panoptic_3d: bool = False,
        poly2mask: bool = True,
        seg_3d_dtype: str = "np.int64",
        seg_offset: int = None,
        dataset_type: str = None,
        backend_args: Optional[dict] = None,
    ) -> None:
        super(LoadScAnnotations3D, self).__init__(
            with_bbox_3d,
            with_label_3d,
            with_attr_label,
            with_mask_3d,
            with_seg_3d,
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            with_bbox_depth,
            with_panoptic_3d,
            poly2mask,
            seg_3d_dtype,
            seg_offset,
            dataset_type,
            backend_args,
        )

    def _load_semantic_seg_3d(self, results: dict) -> dict:
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results["sc_points"]["sc_label_path"]

        try:
            mask_bytes = get(pts_semantic_mask_path, backend_args=self.backend_args)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmengine.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(pts_semantic_mask_path, dtype=np.int64)

        if self.dataset_type == "semantickitti":
            pts_semantic_mask = pts_semantic_mask.astype(np.int64)
            pts_semantic_mask = pts_semantic_mask % self.seg_offset
        # nuScenes loads semantic and panoptic labels from different files.

        results["pts_semantic_mask"] = pts_semantic_mask

        # 'eval_ann_info' will be passed to evaluator
        if "eval_ann_info" in results:
            results["eval_ann_info"]["pts_semantic_mask"] = pts_semantic_mask
        return results

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
            semantic segmentation annotations.
        """
        results = super().transform(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_panoptic_3d:
            results = self._load_panoptic_3d(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)
        return results
