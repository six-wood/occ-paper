# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from torch import Tensor

from mmdet3d.models import EncoderDecoder3D
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList


@MODELS.register_module()
class ScSegmentor(EncoderDecoder3D):
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
        imgs = batch_inputs_dict["imgs"]
        x = self.extract_feat(imgs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, batch_data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, batch_data_samples)
            losses.update(loss_aux)
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

        imgs = batch_inputs_dict["imgs"]
        x = self.extract_feat(imgs)
        seg_labels_list = self.decode_head.predict(x, batch_input_metas, self.test_cfg)

        return self.postprocess_result(seg_labels_list, batch_data_samples)

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

    def postprocess_result(self, seg_labels_list: List[Tensor], batch_data_samples: SampleList) -> SampleList:
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

        for i, seg_pred in enumerate(seg_labels_list):
            batch_data_samples[i].set_data({"pred_pts_seg": PointData(**{"pts_semantic_mask": seg_pred})})
        return batch_data_samples

    # def pack(self, array):
    #     """convert a boolean array into a bitwise array."""
    #     array = array.reshape((-1))

    #     # compressing bit flags.
    #     # yapf: disable
    #     compressed = array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
    #     # yapf: enable

    #     return np.array(compressed, dtype=np.uint8)

    # def get_bev_map(self, voxel_dict, batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
    #     batch_size = len(batch_data_samples)
    #     coors = voxel_dict["coors"]  # z y x
    #     grid_shape = self.data_preprocessor.voxel_layer.grid_shape
    #     gt_shape = batch_data_samples[0].gt_pts_seg.voxel_label.shape
    #     bev_map = torch.zeros(
    #         (batch_size, grid_shape[0], grid_shape[1], grid_shape[2]),
    #         dtype=torch.float32,
    #         device=voxel_dict["voxels"].device,
    #     )  # b x y z
    #     target = torch.zeros(
    #         (batch_size, gt_shape[0], gt_shape[1], gt_shape[2]),
    #         dtype=torch.float32,
    #         device=voxel_dict["voxels"].device,
    #     )  # b x y z
    #     for i in range(batch_size):
    #         coor = coors[coors[:, 0] == i]
    #         bev_map[i, coor[:, 3], coor[:, 2], coor[:, 1]] = 1
    #         target[i] = torch.from_numpy(batch_data_samples[i].gt_pts_seg.voxel_label).cuda()

    #     ones = torch.ones_like(target)
    #     target = torch.where(torch.logical_or(target == self.ignore_index, target == 0), target, ones)

    #     return bev_map, target

    # def loss(self, batch_inputs_dict: Dict[List, Tensor], batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
    #     voxel_dict = batch_inputs_dict["voxels"]
    #     bev_map, target = self.get_bev_map(voxel_dict, batch_data_samples)

    #     out_level_1 = self.forward(bev_map.permute(0, 3, 1, 2).to(target.device))
    #     # calculate loss
    #     losses = dict()
    #     losses_pts = dict()
    #     class_weights_level_1 = self.class_weights_level_1.type_as(target)
    #     loss_sc_level_1 = BCE_ssc_loss(out_level_1, target, class_weights_level_1, self.alpha, self.ignore_index)
    #     losses_pts["loss_sc_level_1"] = loss_sc_level_1

    #     losses.update(losses_pts)

    #     return losses

    # def predict(self, batch_inputs_dict, batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
    #     voxel_dict = batch_inputs_dict["voxels"]
    #     bev_map, target = self.get_bev_map(voxel_dict, batch_data_samples)

    #     sc_pred = self.forward(bev_map.permute(0, 3, 1, 2).to(target.device))
    #     y_pred = sc_pred.detach().cpu().numpy()  # [1, 20, 128, 128, 16]
    #     y_pred = np.argmax(y_pred, axis=1).astype(np.uint8)  # [1, 128, 128, 16]

    #     # save query proposal
    #     # img_path = result["img_path"]
    #     # frame_id = os.path.splitext(img_path[0])[0][-6:]

    #     # msnet3d
    #     # if not os.path.exists(os.path.join("./kitti/dataset/sequences_msnet3d_sweep10", img_metas[0]['sequence_id'], 'queries')):
    #     # os.makedirs(os.path.join("./kitti/dataset/sequences_msnet3d_sweep10", img_metas[0]['sequence_id'], 'queries'))
    #     # save_query_path = os.path.join("./kitti/dataset/sequences_msnet3d_sweep10", img_metas[0]['sequence_id'], 'queries', frame_id + ".query_iou5203_pre7712_rec6153")

    #     # y_pred_bin = self.pack(y_pred)
    #     # y_pred_bin.tofile(save_query_path)
    #     # ---------------------------------------------------------------------------------------------------

    #     result = dict()
    #     y_true = target.cpu().numpy()
    #     result["y_pred"] = y_pred
    #     result["y_true"] = y_true
    #     result = list([result])
    #     return result