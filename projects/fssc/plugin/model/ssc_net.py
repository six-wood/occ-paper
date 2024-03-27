# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence
import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from mmdet3d.models import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet3d.models.utils import add_prefix


@MODELS.register_module()
class SscNet(MVXTwoStageDetector):
    def __init__(
        self,
        bev_backbone: ConfigType = None,
        pts_backbone: ConfigType = None,
        sc_head: ConfigType = None,
        neck: OptConfigType = None,
        ssc_head: ConfigType = None,
        aux_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super(SscNet, self).__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )

        if bev_backbone is not None:
            self.bev_backbone = MODELS.build(bev_backbone)

        if sc_head is not None:
            self.sc_head = MODELS.build(sc_head)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if ssc_head is not None:
            self.ssc_head = MODELS.build(ssc_head)

        if pts_backbone is not None:
            self.pts_backbone = MODELS.build(pts_backbone)

        self.with_aux_head = True if aux_head is not None else False
        if aux_head is not None:
            self.aux_head = MODELS.build(aux_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.grid_shape = self.data_preprocessor.voxel_layer.grid_shape
        self.pc_range = self.data_preprocessor.voxel_layer.point_cloud_range
        self.voxel_size = self.data_preprocessor.voxel_layer.voxel_size

    def extract_bev_feat(self, voxel_dict: Dict[str, Tensor]) -> Tensor:
        """Extract features from BEV images.

        Args:
            imgs (Tensor): BEV images with shape (B, C, H, W).

        Returns:
            Tensor: Extracted features from BEV images.
        """
        coors = voxel_dict["coors"]  # z y x
        batch_size = coors[-1, 0] + 1
        bev_map = torch.zeros(
            (batch_size, self.grid_shape[2], self.grid_shape[0], self.grid_shape[1]),
            dtype=torch.float32,
            device=coors.device,
        )  # channel first(height first)
        bev_map[coors[:, 0], coors[:, 1], coors[:, 3], coors[:, 2]] = 1
        bev_feature = self.bev_backbone(bev_map)  # channel first

        return bev_feature

    def extract_pts_feat(
        self,
        voxel_dict: Dict[str, Tensor],
    ) -> Sequence[Tensor]:
        # feas, coors = self.pts_voxel_encoder(voxel_dict["voxels"], voxel_dict["coors"])
        # feas = self.pts_middle_encoder(feas, coors)
        feas, coors = self.pts_backbone(voxel_dict["voxels"], voxel_dict["num_points"], voxel_dict["coors"])
        return feas, coors

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
        vxoel_dict = batch_inputs_dict["voxels"]

        bev_fea = self.extract_bev_feat(vxoel_dict)
        pts_fea, pts_coors = self.extract_pts_feat(vxoel_dict)
        geo_probs = F.softmax(self.sc_head.predict(bev_fea), dim=1)
        sparse_fea = self.neck(geo_probs, bev_fea, pts_fea, pts_coors)

        losses = dict()

        loss_sc = self.sc_head.loss(bev_fea, batch_data_samples)
        loss_ssc = self.ssc_head.loss(sparse_fea, batch_data_samples)
        if self.with_aux_head:
            loss_aux = self.aux_head.loss(pts_fea, batch_data_samples, self.train_cfg)
            losses.update(add_prefix(loss_aux, "aux"))

        losses.update(add_prefix(loss_sc, "sc"))
        losses.update(add_prefix(loss_ssc, "ssc"))

        # torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated() / 1024 / 1024 / 1024, "GB")
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

        vxoel_dict = batch_inputs_dict["voxels"]

        bev_fea = self.extract_bev_feat(vxoel_dict)
        pts_fea, pts_coors = self.extract_pts_feat(vxoel_dict)
        geo_probs = F.softmax(self.sc_head.predict(bev_fea), dim=1)
        sparse_fea = self.neck(geo_probs, bev_fea, pts_fea, pts_coors)

        ssc_pred = self.ssc_head.predict(sparse_fea).argmax(dim=1)

        return self.postprocess_result(ssc_pred, sparse_fea.indices, batch_data_samples)

    def postprocess_result(self, ssc_labels: Tensor, sc_coors: Tensor, batch_data_samples: SampleList) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Args:
            coors: b z y x

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """
        ssc_true = np.stack([data_sample.metainfo["voxel_label"] for data_sample in batch_data_samples], axis=0)
        B, H, W, D = ssc_true.shape
        ssc_pred = torch.zeros((B, H, W, D), dtype=torch.long, device=ssc_labels.device)
        ssc_pred[sc_coors[:, 0], sc_coors[:, 3], sc_coors[:, 2], sc_coors[:, 1]] = ssc_labels
        ssc_pred = ssc_pred.cpu().numpy()

        for i, batch_data in enumerate(batch_data_samples):
            batch_data.set_data({"y_pred": ssc_pred[i]})
            batch_data.set_data({"y_true": ssc_true[i]})
        return batch_data_samples
