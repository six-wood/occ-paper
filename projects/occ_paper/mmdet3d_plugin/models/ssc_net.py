import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch._tensor import Tensor
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.structures import Det3DDataSample
from typing import Dict, List, Optional, Sequence, Union
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.models.utils import add_prefix
from .utils import compute_visibility_mask


@MODELS.register_module()
class SscNet(MVXTwoStageDetector):
    def __init__(
        self,
        use_pred_mask: bool = False,
        img_backbone: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_bev_backbone: Optional[dict] = None,
        pts_range_backbone: Optional[dict] = None,
        fusion_neck: Optional[dict] = None,
        ssc_head: Optional[dict] = None,
        auxiliary_head: OptMultiConfig = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            # img_backbone=img_backbone,
            # img_neck=img_neck,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor,
        )
        """
        SSCNet architecture
        :param N: number of classes to be predicted (i.e. 12 for NYUv2)
        """
        if pts_bev_backbone is not None:
            self.pts_bev_backbone = MODELS.build(pts_bev_backbone)
        if pts_range_backbone is not None:
            self.pts_range_backbone = MODELS.build(pts_range_backbone)
        if fusion_neck is not None:
            self.fusion_neck = MODELS.build(fusion_neck)
        if ssc_head is not None:
            self.ssc_head = MODELS.build(ssc_head)

        self.with_auxiliary_head = True if auxiliary_head is not None else False
        self._init_auxiliary_head(auxiliary_head)

        self.use_pred_mask = use_pred_mask
        self.grid_shape = self.data_preprocessor.voxel_layer.grid_shape
        self.pc_range = self.data_preprocessor.voxel_layer.point_cloud_range
        self.voxel_size = self.data_preprocessor.voxel_layer.voxel_size
        self.fov = [self.data_preprocessor.range_layer.fov_down, self.data_preprocessor.range_layer.fov_up]
        if self.use_pred_mask:
            self.visibility_mask = compute_visibility_mask(
                center=[0, 0, 0],
                pc_range=self.pc_range,
                voxel_size=self.voxel_size,
                fov=self.fov,
            )

    def _init_auxiliary_head(self, auxiliary_head: OptMultiConfig = None) -> None:
        """Initialize ``auxiliary_head``."""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def _auxiliary_head_forward_train(
        self,
        batch_inputs_dict: dict,
        batch_data_samples: SampleList,
    ) -> Dict[str, Tensor]:
        """Run forward function and calculate loss for auxiliary head in
        training.

        Args:
            batch_input (Tensor): Input point cloud sample
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components for auxiliary
            head.
        """
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(batch_inputs_dict, batch_data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f"aux_{idx}"))
        else:
            loss_aux = self.auxiliary_head.loss(batch_inputs_dict, batch_data_samples, self.train_cfg)
            losses.update(add_prefix(loss_aux, "aux"))

        return losses

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
        range_features = self.pts_range_backbone(range_dict["range_imgs"])
        geo_fea, sem_fea, sc_query_grid_coor = self.fusion_neck(voxel_features, range_features)  # channel first
        return geo_fea, sem_fea, sc_query_grid_coor

    def loss(self, batch_inputs_dict: Dict[List, Tensor], batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        # imgs = batch_inputs_dict.get("imgs", None)
        voxel_dict = batch_inputs_dict.get("voxels", None)
        range_dict = batch_inputs_dict.get("range_imgs", None)
        # batch_input_metas = [item.metainfo for item in batch_data_samples]

        geo_fea, sem_fea, sc_query_grid_coor = self.extract_pts_feat(voxel_dict, range_dict, batch_data_samples)
        # img_fea = self.extract_img_feat(imgs, batch_input_metas)
        losses = self.ssc_head.loss(geo_fea, sem_fea, sc_query_grid_coor, batch_data_samples, self.train_cfg)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, batch_data_samples)
            losses.update(loss_aux)
        return losses

    def predict(self, batch_inputs_dict, batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        # imgs = batch_inputs_dict.get("imgs", None)
        voxel_dict = batch_inputs_dict.get("voxels", None)
        range_dict = batch_inputs_dict.get("range_imgs", None)
        # batch_input_metas = [item.metainfo for item in batch_data_samples]

        geo_fea, sem_fea, sc_query_grid_coor = self.extract_pts_feat(voxel_dict, range_dict, batch_data_samples)
        # img_fea = self.extract_img_feat(imgs, batch_input_metas)
        geo_logits, sem_logits = self.ssc_head.predict(geo_fea, sem_fea, sc_query_grid_coor, batch_data_samples)
        results = self.postprocess_result(geo_logits, sem_logits, sc_query_grid_coor, batch_data_samples)
        return results

    def postprocess_result(self, geo_logits: Tensor, sem_logits: Tensor, sc_query_grid_coor: Tensor, batch_data_samples):
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

        gt_semantic_segs = [data_sample.gt_pts_seg.voxel_label.long() for data_sample in batch_data_samples]
        seg_true = torch.stack(gt_semantic_segs, dim=0).cpu().numpy()

        B = seg_true.shape[0]
        unknown_idx = sem_logits.shape[1]
        pred = unknown_idx * geo_logits.argmax(dim=1)
        sem_pred = sem_logits.argmax(dim=1)
        pred[sc_query_grid_coor[:, :, 0], sc_query_grid_coor[:, :, 1], sc_query_grid_coor[:, :, 2], sc_query_grid_coor[:, :, 3]] = sem_pred
        pred = pred.cpu().numpy()

        if self.use_pred_mask:
            pred = np.where(self.visibility_mask[None, :], pred, 0)
        result = dict()
        result["y_pred"] = pred
        result["y_true"] = seg_true
        result = list([result])
        return result

    def occupied_voxels_to_pc(self, occupancy_grid):
        """Saves only occupied voxels to a text file."""
        # Reshape the grid
        grid_shape = self.grid_shape
        x_scale, y_scale, z_scale = self.voxel_size
        x_offset, y_offset, z_offset = self.pc_range[:3]

        reshaped_grid = occupancy_grid.reshape(grid_shape)

        # Generate grid coordinates
        x, y, z = np.meshgrid(np.arange(grid_shape[0]), np.arange(grid_shape[1]), np.arange(grid_shape[2]), indexing="ij")
        x = x * x_scale + x_offset
        y = y * y_scale + y_offset
        z = z * z_scale + z_offset

        # Flatten and filter out unoccupied voxels
        coordinates = np.vstack((x.ravel(), y.ravel(), z.ravel(), reshaped_grid.ravel())).T
        occupied_coordinates = coordinates[coordinates[:, 3] > 0]
        return occupied_coordinates

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        pass
        # data = self.data_preprocessor(data, False)
        # batch_inputs_dict = data["inputs"]
        # batch_data_samples = data["data_samples"]
        # voxel_dict = batch_inputs_dict["voxels"]
        # range_dict = batch_inputs_dict["range_imgs"]

        # pts_fea = self.extract_pts_feat(voxel_dict, range_dict, batch_data_samples)
        # seg_logits = self.ssc_head.predict(pts_fea, batch_data_samples)
        # seg_pred = seg_logits.argmax(dim=1).cpu().numpy()
        # if self.use_pred_mask:
        #     seg_pred = np.where(self.visibility_mask[None, :], seg_pred, 0)
        # pred_pc = self.occupied_voxels_to_pc(seg_pred)
        # pred_sem = pred_pc[:, 3].astype(np.int32)
        # pred_geo = pred_pc[:, :3]

        # for data_sample in batch_data_samples:
        #     data_sample.set_data(
        #         {
        #             "pred_pts_geo": pred_geo,
        #             "pred_pts_seg": PointData(**{"pts_semantic_mask": pred_sem}),
        #         }
        #     )
        # return batch_data_samples
