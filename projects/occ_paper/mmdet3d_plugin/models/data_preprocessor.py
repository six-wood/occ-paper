# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import List, Optional, Sequence, Union
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models.utils.misc import samplelist_boxtype2tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import OptConfigType
from mmdet3d.models.data_preprocessors.data_preprocessor import Det3DDataPreprocessor


class SemkittiRangeView:
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
        self.fov_up = fov_up / 180.0 * torch.pi
        self.fov_down = fov_down / 180.0 * torch.pi
        self.fov_left = fov_left / 180.0 * torch.pi
        self.fov_right = fov_right / 180.0 * torch.pi
        self.fov_y = abs(self.fov_down) + abs(self.fov_up)
        self.fov_x = abs(self.fov_right) + abs(self.fov_left)
        self.means = torch.Tensor(means)
        self.stds = torch.Tensor(stds)
        self.ignore_index = ignore_index

    def ranglize(self, input: list):
        range_dict = {}
        range_dict["range_imgs"] = torch.cat([self.transform(res).unsqueeze(0).permute(0, 3, 1, 2) for res in input], dim=0)
        return range_dict

    def transform(self, points: torch.Tensor) -> dict:
        device = points.device
        self.means = self.means.to(device)
        self.stds = self.stds.to(device)
        proj_image = torch.full((self.H, self.W, 5), -1, dtype=torch.float32, device=device)
        proj_idx = torch.full((self.H, self.W), -1, dtype=torch.int32, device=device)

        # get depth of all points
        depth = torch.norm(points[:, :3], p=2, dim=1)

        # get angles of all points
        yaw = -torch.arctan2(points[:, 1], points[:, 0])
        pitch = torch.arcsin(points[:, 2] / depth)

        # filter based on angles
        fov_in = torch.logical_and(
            torch.logical_and((pitch < self.fov_up), (pitch > self.fov_down)),
            torch.logical_and((yaw < self.fov_right), (yaw > self.fov_left)),
        )

        # get projection in image coords
        proj_x = (yaw + abs(self.fov_left)) / self.fov_x
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov_y

        # scale to image size using angular resolution
        proj_x *= self.W
        proj_y *= self.H

        proj_x = proj_x[fov_in]
        proj_y = proj_y[fov_in]
        depth = depth[fov_in]

        # round and clamp for use as index
        proj_x = torch.floor(proj_x)
        proj_x = torch.clamp(proj_x, 0, self.W - 1).to(torch.int32)

        proj_y = torch.floor(proj_y)
        proj_y = torch.clamp(proj_y, 0, self.H - 1).to(torch.int32)

        # order in decreasing depth
        indices = torch.arange(depth.shape[0], device=device).to(torch.int32)
        order = torch.argsort(depth, descending=True).to(torch.int32)
        proj_idx[proj_y[order], proj_x[order]] = indices[order]
        proj_image[proj_y[order], proj_x[order], 0] = depth[order]

        # depth_img = proj_image[:, :, 0].detach().cpu().numpy()
        # cv2.normalize(depth_img, depth_img, 0, 255, cv2.NORM_MINMAX)
        # cv2.imwrite("proj_image.bmp", depth_img)

        proj_image[proj_y[order], proj_x[order], 1:] = points[order]
        proj_mask = (proj_idx > 0).to(torch.int32)

        proj_image = (proj_image - self.means[None, None, :]) / self.stds[None, None, :]
        proj_image = proj_image * proj_mask[..., None].to(torch.float32)

        return proj_image

    def get_range_view_coord(self, points: torch.Tensor) -> dict:
        # get depth of all points
        depth = torch.norm(points[:, :, :3], p=2, dim=-1)

        # get angles of all points
        yaw = -torch.arctan2(points[:, :, 1], points[:, :, 0])
        pitch = torch.arcsin(points[:, :, 2] / depth)

        # get projection in image coords
        proj_x = (yaw + abs(self.fov_left)) / self.fov_x
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov_y

        # scale to image size using angular resolution
        proj_x *= self.W
        proj_y *= self.H
        return torch.stack([proj_y, proj_x], dim=-1)


@MODELS.register_module()
class SccDataPreprocessor(Det3DDataPreprocessor):
    """Points / Image pre-processor for point clouds / vision-only / multi-
    modality 3D detection tasks.

    It provides the data pre-processing as follows

    - Collate and move image and point cloud data to the target device.

    - 1) For image data:

      - Pad images in inputs to the maximum size of current batch with defined
        ``pad_value``. The padding size can be divisible by a defined
        ``pad_size_divisor``.
      - Stack images in inputs to batch_imgs.
      - Convert images in inputs from bgr to rgb if the shape of input is
        (3, H, W).
      - Normalize images in inputs with defined std and mean.
      - Do batch augmentations during training.

    - 2) For point cloud data:

      - If no voxelization, directly return list of point cloud data.
      - If voxelization is applied, voxelize point cloud according to
        ``voxel_type`` and obtain ``voxels``.

    Args:
        voxel (bool): Whether to apply voxelization to point cloud.
            Defaults to False.
        voxel_type (str): Voxelization type. Two voxelization types are
            provided: 'hard' and 'dynamic', respectively for hard voxelization
            and dynamic voxelization. Defaults to 'hard'.
        voxel_layer (dict or :obj:`ConfigDict`, optional): Voxelization layer
            config. Defaults to None.
        batch_first (bool): Whether to put the batch dimension to the first
            dimension when getting voxel coordinates. Defaults to True.
        max_voxels (int, optional): Maximum number of voxels in each voxel
            grid. Defaults to None.
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be divisible by
            ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic segmentation
            maps. Defaults to 255.
        bgr_to_rgb (bool): Whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): Whether to convert image from RGB to BGR.
            Defaults to False.
        boxtype2tensor (bool): Whether to convert the ``BaseBoxes`` type of
            bboxes data to ``Tensor`` type. Defaults to True.
        non_blocking (bool): Whether to block current process when transferring
            data to device. Defaults to False.
        batch_augments (List[dict], optional): Batch-level augmentations.
            Defaults to None.
    """

    def __init__(
        self,
        range_img: bool = False,
        range_layer: dict = None,
        voxel: bool = False,
        voxel_type: str = "hard",
        voxel_layer: OptConfigType = None,
        batch_first: bool = True,
        max_voxels: Optional[int] = None,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        pad_mask: bool = False,
        mask_pad_value: int = 0,
        pad_seg: bool = False,
        seg_pad_value: int = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        boxtype2tensor: bool = True,
        non_blocking: bool = False,
        batch_augments: Optional[List[dict]] = None,
    ) -> None:
        super().__init__(
            voxel,
            voxel_type,
            voxel_layer,
            batch_first,
            max_voxels,
            mean,
            std,
            pad_size_divisor,
            pad_value,
            pad_mask,
            mask_pad_value,
            pad_seg,
            seg_pad_value,
            bgr_to_rgb,
            rgb_to_bgr,
            boxtype2tensor,
            non_blocking,
            batch_augments,
        )
        self.range_img = range_img
        self.range_layer = range_layer
        if self.range_img:
            self.ranglize = SemkittiRangeView(**self.range_layer)

    def simple_process(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion for img data
        based on ``BaseDataPreprocessor``, and voxelize point cloud if `voxel`
        is set to be True.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        if "img" in data["inputs"]:
            batch_pad_shape = self._get_pad_shape(data)

        data = self.collate_data(data)
        inputs, data_samples = data["inputs"], data["data_samples"]
        batch_inputs = dict()

        if "points" in inputs:
            batch_inputs["points"] = inputs["points"]

            if self.voxel:
                voxel_dict = self.voxelize(inputs["points"], data_samples)
                batch_inputs["voxels"] = voxel_dict

            if self.range_img:
                range_dict = self.ranglize.ranglize(inputs["points"])
                batch_inputs["range_imgs"] = range_dict

        if "imgs" in inputs:
            imgs = inputs["imgs"]

            if data_samples is not None:
                # NOTE the batched image size information may be useful, e.g.
                # in DETR, this is needed for the construction of masks, which
                # is then used for the transformer_head.
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                    data_sample.set_metainfo({"batch_input_shape": batch_input_shape, "pad_shape": pad_shape})

                if self.boxtype2tensor:
                    samplelist_boxtype2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)
                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs["imgs"] = imgs

        return {"inputs": batch_inputs, "data_samples": data_samples}
