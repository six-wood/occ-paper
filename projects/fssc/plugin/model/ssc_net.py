# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List
import torch
from torch import Tensor
import numpy as np

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
        sc_head: ConfigType = None,
        # neck: OptConfigType = None,
        # sparse_backbone: OptConfigType = None,
        # ssc_head: OptConfigType = None,
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

        # if neck is not None:
        #     self.neck = MODELS.build(neck)

        # if sparse_backbone is not None:
        #     self.sparse_backbone = MODELS.build(sparse_backbone)
        # if ssc_head is not None:
        #     self.ssc_head = MODELS.build(ssc_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

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

        # geo_pred = self.sc_head.predict(y).argmax(dim=1)
        # sem_fea, coors = self.neck(y, geo_pred)
        # sparse_fea = self.sparse_backbone(sem_fea, coors)

        # sparse_dict = {"sem_fea": sparse_fea, "coors": coors}

        losses = dict()

        loss_sc = self.sc_head.loss(y, batch_data_samples)
        # loss_ssc = self.ssc_head.loss(sparse_dict, batch_data_samples, self.train_cfg)

        losses.update(add_prefix(loss_sc, "sc"))
        # losses.update(add_prefix(loss_ssc, "ssc"))

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
        # 3D segmentation requires per-point prediction, so it's impossible
        # to use down-sampling to get a batch of scenes with same num_points
        # therefore, we only support testing one scene every time
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)

        vxoels = batch_inputs_dict["voxels"]
        y = self.extract_bev_feat(vxoels)

        geo_pred = self.sc_head.predict(y).argmax(dim=1)
        # sem_fea, coors = self.neck(y, geo_pred)
        # sem_fea = self.sparse_backbone(sem_fea, coors)

        # ssc_labels = self.ssc_head.predict(sem_fea, batch_data_samples).argmax(dim=1)

        return self.postprocess_result(geo_pred, batch_data_samples)

    def postprocess_result(self, ssc_labels: List[Tensor], batch_data_samples: SampleList) -> SampleList:
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
        ssc_pred = ssc_labels.cpu().numpy()

        for i, batch_data in enumerate(batch_data_samples):
            batch_data.set_data({"y_pred": ssc_pred[i]})
            batch_data.set_data({"y_true": ssc_true[i]})
        return batch_data_samples

    def _forward(self, batch_inputs_dict: dict, batch_data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        imgs = batch_inputs_dict["imgs"]
        x = self.extract_feat(imgs)
        return self.decode_head.forward(x)


palette = list(
    [
        [0, 0, 0],
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
    ]
)
