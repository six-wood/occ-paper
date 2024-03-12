# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List
import torch
from torch import Tensor
import numpy as np

from mmdet3d.models import EncoderDecoder3D
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class RangeImageSegmentor(EncoderDecoder3D):
    def __init__(
        self,
        backbone: ConfigType = None,
        bev_backbone: ConfigType = None,
        decode_head: ConfigType = None,
        sc_head: ConfigType = None,
        neck: OptConfigType = None,
        auxiliary_head: OptMultiConfig = None,
        loss_regularization: OptMultiConfig = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super(RangeImageSegmentor, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            loss_regularization=loss_regularization,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
        if bev_backbone is not None:
            self.bev_backbone = MODELS.build(bev_backbone)
        if sc_head is not None:
            self.sc_head = MODELS.build(sc_head)

        self.grid_shape = self.data_preprocessor.voxel_layer.grid_shape
        self.pc_range = self.data_preprocessor.voxel_layer.point_cloud_range
        self.voxel_size = self.data_preprocessor.voxel_layer.voxel_size

    def extract_bev_feat(self, voxels: Tensor) -> Tensor:
        """Extract features from BEV images.

        Args:
            imgs (Tensor): BEV images with shape (B, C, H, W).

        Returns:
            Tensor: Extracted features from BEV images.
        """
        coors = voxels["coors"]  # z y x
        batch_size = coors[-1, 0] + 1
        bev_map = torch.zeros(
            (batch_size, self.grid_shape[2], self.grid_shape[0], self.grid_shape[1]),
            dtype=torch.float32,
            device=coors.device,
        )  # channel first(height first)
        bev_map[coors[:, 0], coors[:, 1], coors[:, 3], coors[:, 2]] = 1
        bev_feature = self.bev_backbone(bev_map)  # channel first

        return bev_feature

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        # extract features using backbone
        vxoels = batch_inputs_dict["voxels"]
        y = self.extract_bev_feat(vxoels)

        losses = dict()

        loss_geo = self.sc_head.loss(y, batch_data_samples)

        losses.update(loss_geo)

        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList, rescale: bool = True) -> SampleList:
        """Simple test with single scene.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """
        # 3D segmentation requires per-point prediction, so it's impossible
        # to use down-sampling to get a batch of scenes with same num_points
        # therefore, we only support testing one scene every time
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)

        vxoels = batch_inputs_dict["voxels"]
        y = self.extract_bev_feat(vxoels)

        geo_labels = self.sc_head.predict(y)

        return self.postprocess_result(geo_labels, batch_data_samples)

    def postprocess_result(self, geo_labels: Tensor, batch_data_samples: SampleList) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Args:
            seg_labels_list (List[Tensor]): List of segmentation results,
                seg_logits from model of each input point clouds sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """

        sc_true = np.stack([data_sample.metainfo["voxel_label"] for data_sample in batch_data_samples], axis=0)
        sc_pred = geo_labels.cpu().numpy()
        for i, batch_data in enumerate(batch_data_samples):
            batch_data.set_data({"y_pred": sc_pred[i]})
            batch_data.set_data({"y_true": sc_true[i]})
        return batch_data_samples
