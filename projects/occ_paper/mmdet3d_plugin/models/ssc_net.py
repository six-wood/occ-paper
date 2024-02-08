import os
from typing import Dict, List, Optional, Sequence
import torch
import numpy as np
from torch import Tensor
from torch._tensor import Tensor
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmdet3d.structures import Det3DDataSample
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmcv.cnn import ConvModule, build_activation_layer, build_conv_layer, build_norm_layer
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class SscNet(MVXTwoStageDetector):
    def __init__(
        self,
        pts_bev_backbone: Optional[dict] = None,
        pts_range_backbone: Optional[dict] = None,
        pts_fusion_neck: Optional[dict] = None,
        pts_ssc_head: Optional[dict] = None,
        pts_ssc_loss: Optional[dict] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(train_cfg=train_cfg, test_cfg=test_cfg, init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        """
        SSCNet architecture
        :param N: number of classes to be predicted (i.e. 12 for NYUv2)
        """
        self.fuse_cfg = pts_fusion_neck
        if pts_bev_backbone is not None:
            self.pts_bev_backbone = MODELS.build(pts_bev_backbone)
        if pts_range_backbone is not None:
            self.pts_range_backbone = MODELS.build(pts_range_backbone)
        if pts_fusion_neck is not None:
            self.pts_fusion_neck = MODELS.build(pts_fusion_neck)
        if pts_ssc_head is not None:
            self.pts_ssc_head = MODELS.build(pts_ssc_head)
        if pts_ssc_loss is not None:
            self.pts_scc_loss = MODELS.build(pts_ssc_loss)

        self.grid_shape = self.data_preprocessor.voxel_layer.grid_shape

    def extract_pts_feat(self, voxel_dict: Dict[str, Tensor], range_dict: Dict[str, Tensor], batch_data_samples) -> Sequence[Tensor]:
        """Extract features of points.
        All Channel first.
        Args:
            voxel_dict(Dict[str, Tensor]): Dict of voxelization infos.
            points (List[tensor], optional):  Point cloud of multiple inputs.
            img_feats (list[Tensor], tuple[tensor], optional): Features from
                image backbone.
            batch_input_metas (list[dict], optional): The meta information
                of multiple samples. Defaults to True.

        Returns:
            Sequence[tensor]: points features of multiple inputs
            from backbone or neck.
        """
        coors = voxel_dict["coors"]  # z y x
        batch_size = coors[-1, 0] + 1
        bev_map = torch.zeros(
            (batch_size, self.grid_shape[2], self.grid_shape[0], self.grid_shape[1]),
            dtype=torch.float32,
            device=coors.device,
        )  # channel first(height first)
        bev_map[coors[:, 0], coors[:, 1], coors[:, 3], coors[:, 2]] = 1
        voxel_features = self.pts_bev_backbone(bev_map)  # channel first
        if self.fuse_cfg is None:
            return voxel_features
        range_features = self.pts_range_backbone(range_dict["range_imgs"])
        fuse_fea = self.pts_fusion_neck(voxel_features, range_features)  # channel first
        return fuse_fea

    def loss(self, batch_inputs_dict: Dict[List, Tensor], batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        voxel_dict = batch_inputs_dict["voxels"]
        range_dict = batch_inputs_dict["range_imgs"]

        pts_fea = self.extract_pts_feat(voxel_dict, range_dict, batch_data_samples)
        losses = self.pts_ssc_head.loss(pts_fea, batch_data_samples, self.train_cfg)
        return losses

    def predict(self, batch_inputs_dict, batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        voxel_dict = batch_inputs_dict["voxels"]
        range_dict = batch_inputs_dict["range_imgs"]

        pts_fea = self.extract_pts_feat(voxel_dict, range_dict, batch_data_samples)
        seg_logits = self.pts_ssc_head.predict(pts_fea, batch_data_samples)
        results = self.postprocess_result(seg_logits, batch_data_samples)
        return results

    def postprocess_result(self, seg_logits: List[Tensor], batch_data_samples):
        """Convert results list to `Det3DDataSample`.

        Args:
            seg_logits_list (List[Tensor]): List of segmentation results,
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
        assert len(batch_data_samples) == 1  # only support batch_size=1
        seg_pred = seg_logits.argmax(dim=1).cpu().numpy()

        gt_semantic_segs = [data_sample.gt_pts_seg.voxel_label.long() for data_sample in batch_data_samples]
        seg_true = torch.stack(gt_semantic_segs, dim=0).cpu().numpy()
        result = dict()
        result["y_pred"] = seg_pred
        result["y_true"] = seg_true
        result = list([result])
        return result
