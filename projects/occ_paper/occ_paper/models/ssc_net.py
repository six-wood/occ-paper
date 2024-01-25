import os
from typing import Dict, List, Optional
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
        task: str = "sc",
        ignore_index: int = 255,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        pts_fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        pts_bbox_head: Optional[dict] = None,
        pts_scc_head: Optional[dict] = None,
        pts_scc_loss: Optional[dict] = None,
        img_roi_head: Optional[dict] = None,
        img_rpn_head: Optional[dict] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            init_cfg,
            data_preprocessor,
            **kwargs
        )
        """
        SSCNet architecture
        :param N: number of classes to be predicted (i.e. 12 for NYUv2)
        """
        if pts_scc_head is not None:
            self.pts_scc_head = MODELS.build(pts_scc_head)
        if pts_scc_loss is not None:
            self.pts_scc_loss = MODELS.build(pts_scc_loss)
        self.ignore_index = ignore_index
        self.task = task

    def get_bev_map(self, voxel_dict, batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        batch_size = len(batch_data_samples)
        coors = voxel_dict["coors"]  # z y x
        grid_shape = self.data_preprocessor.voxel_layer.grid_shape
        gt_shape = batch_data_samples[0].gt_pts_seg.voxel_label.shape
        bev_map = torch.zeros(
            (batch_size, grid_shape[0], grid_shape[1], grid_shape[2]),
            dtype=torch.float32,
            device=voxel_dict["voxels"].device,
        )  # b x y z
        target = torch.zeros(
            (batch_size, gt_shape[0], gt_shape[1], gt_shape[2]),
            dtype=torch.float32,
            device=voxel_dict["voxels"].device,
        )  # b x y z
        for i in range(batch_size):
            coor = coors[coors[:, 0] == i]
            bev_map[i, coor[:, 3], coor[:, 2], coor[:, 1]] = 1
            target[i] = torch.from_numpy(batch_data_samples[i].gt_pts_seg.voxel_label).cuda()

        if self.task == "sc":
            ones = torch.ones_like(target)
            target = torch.where(torch.logical_or(target == self.ignore_index, target == 0), target, ones)

        return bev_map, target

    def loss(self, batch_inputs_dict: Dict[List, Tensor], batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        voxel_dict = batch_inputs_dict["voxels"]
        bev_map, target = self.get_bev_map(voxel_dict, batch_data_samples)

        pts_fea = self.pts_backbone(bev_map.permute(0, 3, 1, 2).to(target.device))
        pred = self.pts_scc_head(pts_fea)

        # calculate loss
        losses = dict()
        losses_pts = dict()
        loss_sc = self.pts_scc_loss(pred, target)
        losses_pts["loss_sc"] = loss_sc

        losses.update(losses_pts)

        return losses

    def predict(self, batch_inputs_dict, batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        voxel_dict = batch_inputs_dict["voxels"]
        bev_map, target = self.get_bev_map(voxel_dict, batch_data_samples)

        pts_fea = self.pts_backbone(bev_map.permute(0, 3, 1, 2).to(target.device))
        sc_pred = self.pts_scc_head(pts_fea)
        y_pred = sc_pred.detach().cpu().numpy()  # [1, 20, 128, 128, 16]
        y_pred = np.argmax(y_pred, axis=1).astype(np.uint8)

        result = dict()
        y_true = target.cpu().numpy()
        result["y_pred"] = y_pred
        result["y_true"] = y_true
        result = list([result])
        return result
