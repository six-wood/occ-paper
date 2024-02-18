import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from torch._tensor import Tensor
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.structures import Det3DDataSample
from typing import Dict, List, Optional, Sequence, Union
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector


def compute_visibility_mask(
    center: list = [0, 0, 0],
    pc_range: list = [0, -25.6, -2.0, 51.2, 25.6, 4.4],
    voxel_size: list = [0.2, 0.2, 0.2],
    fov: list = [-25.0, 3.0],
) -> np.ndarray:
    # 计算网格大小
    pc_range = np.array(pc_range)
    voxel_size = np.array(voxel_size)
    fov = np.array(fov)
    grid_size = np.round((pc_range[3:] - pc_range[:3]) / voxel_size).astype(np.int32)

    # 确定每个轴的范围
    x_range = np.linspace(pc_range[0] + voxel_size[0] / 2, pc_range[3] - voxel_size[0] / 2, grid_size[0])
    y_range = np.linspace(pc_range[1] + voxel_size[1] / 2, pc_range[4] - voxel_size[1] / 2, grid_size[1])
    z_range = np.linspace(pc_range[2] + voxel_size[2] / 2, pc_range[5] - voxel_size[2] / 2, grid_size[2])

    # 生成三维网格
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing="ij")

    # 调整网格以反映中心点的偏移
    xx -= center[0]
    yy -= center[1]
    zz -= center[2]

    # 计算每个点的俯仰角
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    pitch_angles = np.arcsin(zz / r)

    # 转换为度
    pitch_angles_degrees = np.degrees(pitch_angles)

    # 确定每个体素是否在视场范围内
    visibility_mask = (pitch_angles_degrees >= fov[0]) & (pitch_angles_degrees <= fov[1])

    return visibility_mask


@MODELS.register_module()
class SscNet(MVXTwoStageDetector):
    def __init__(
        self,
        use_pred_mask: bool = False,
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
        super().__init__(
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor,
        )
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

        self.use_pred_mask = use_pred_mask
        self.grid_shape = self.data_preprocessor.voxel_layer.grid_shape
        self.pc_range = self.data_preprocessor.voxel_layer.point_cloud_range
        self.voxel_size = self.data_preprocessor.voxel_layer.voxel_size
        self.fov = [self.data_preprocessor.range_layer.fov_down, self.data_preprocessor.range_layer.fov_up]
        if self.use_pred_mask:
            self.mask = compute_visibility_mask(
                center=[0, 0, 0],
                pc_range=self.pc_range,
                voxel_size=self.voxel_size,
                fov=self.fov,
            )

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
        seg_pred = seg_logits.argmax(dim=1).cpu().numpy()

        gt_semantic_segs = [data_sample.gt_pts_seg.voxel_label.long() for data_sample in batch_data_samples]
        seg_true = torch.stack(gt_semantic_segs, dim=0).cpu().numpy()
        if self.use_pred_mask:
            seg_pred[:, ~self.mask] = 0
        result = dict()
        result["y_pred"] = seg_pred
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
        data = self.data_preprocessor(data, False)
        batch_inputs_dict = data["inputs"]
        batch_data_samples = data["data_samples"]
        voxel_dict = batch_inputs_dict["voxels"]
        range_dict = batch_inputs_dict["range_imgs"]

        pts_fea = self.extract_pts_feat(voxel_dict, range_dict, batch_data_samples)
        seg_logits = self.pts_ssc_head.predict(pts_fea, batch_data_samples)
        seg_pred = seg_logits.argmax(dim=1).cpu().numpy()
        if self.use_pred_mask:
            seg_pred[:, ~self.mask] = 0
        pred_pc = self.occupied_voxels_to_pc(seg_pred)
        pred_sem = pred_pc[:, 3].astype(np.int32)
        pred_geo = pred_pc[:, :3]

        for data_sample in batch_data_samples:
            data_sample.set_data(
                {
                    "pred_pts_geo": pred_geo,
                    "pred_pts_seg": PointData(**{"pts_semantic_mask": pred_sem}),
                }
            )
        return batch_data_samples
