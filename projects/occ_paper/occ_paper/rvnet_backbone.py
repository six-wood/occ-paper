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

from .ssc_loss import BCE_ssc_loss


@MODELS.register_module()
class LMSCNet_SS(MVXTwoStageDetector):
    def __init__(
        self,
        input_dimensions=None,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN"),
        act_cfg: ConfigType = dict(type="ReLU"),
        class_num=None,
        out_scale=None,
        gamma=0,
        alpha=0.5,
        ignore_index=255,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        pts_fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        pts_bbox_head: Optional[dict] = None,
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

        # model parameters
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.out_scale = out_scale
        self.nbr_classes = class_num
        self.gamma = gamma
        self.alpha = alpha
        self.input_dimensions = input_dimensions  # Grid dimensions should be (W, H, D).. z or height being axis 1
        self.ignore_index = ignore_index
        f = self.input_dimensions[1]  # Height of the input grid, H to channels

        self.pool = nn.MaxPool2d(2)  # [F=2; S=2; P=0; D=1]

        self.pooling = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.class_frequencies_level1 = np.array([5.41773033e09, 4.03113667e08])

        self.class_weights_level_1 = torch.from_numpy(1 / np.log(self.class_frequencies_level1 + 0.001))

        self.Encoder_block1 = self._make_conv_layer(f, f)
        self.Encoder_block2 = self._make_encoder_layer(f, int(f * 1.5))
        self.Encoder_block3 = self._make_encoder_layer(int(f * 1.5), int(f * 2))
        self.Encoder_block4 = self._make_encoder_layer(int(f * 2), int(f * 2.5))

        self.Decoder_block1 = self._make_decoder_layer(int(f * 2.5), int(f * 2))
        self.Decoder_block2 = self._make_decoder_layer(int(f * 4), int(f * 1.5))
        self.Decoder_block3 = self._make_decoder_layer(int(f * 3), f)
        self.Decoder_block4 = self._make_conv_layer(int(f * 2), f)

        self.up_sample1 = nn.ConvTranspose2d(int(f * 2), int(f * 2), kernel_size=4, padding=0, stride=4, bias=False)
        self.up_sample2 = nn.ConvTranspose2d(int(f * 1.5), int(f * 1.5), kernel_size=6, padding=2, stride=2, bias=False)

        self.fuse_block = self._make_fuse_layer(int(f * 2) + int(f * 1.5) + int(f) + int(f), f)

        self.seg_head = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])

    def _make_conv_layer(self, in_channels: int, out_channels: int) -> None:  # two conv blocks in beginning
        return nn.Sequential(
            build_conv_layer(self.conv_cfg, in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_activation_layer(self.act_cfg),
            build_conv_layer(self.conv_cfg, out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_activation_layer(self.act_cfg),
        )

    def _make_encoder_layer(self, in_channels: int, out_channels: int) -> None:  # create encoder blocks
        return nn.Sequential(
            nn.MaxPool2d(2),
            build_conv_layer(self.conv_cfg, in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_activation_layer(self.act_cfg),
            build_conv_layer(self.conv_cfg, out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_activation_layer(self.act_cfg),
        )

    def _make_decoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            build_conv_layer(self.conv_cfg, out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_activation_layer(self.act_cfg),
            build_conv_layer(self.conv_cfg, out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_activation_layer(self.act_cfg),
        )

    def _make_fuse_layer(self, in_channels, out_channels):
        return nn.Sequential(
            build_conv_layer(self.conv_cfg, in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_activation_layer(self.act_cfg),
        )

    def step(self, x):
        # Encoder
        enc1 = self.Encoder_block1(x)  # [bs, 32, 256, 256]
        enc2 = self.Encoder_block2(enc1)  # [bs, 48, 128, 128]
        enc3 = self.Encoder_block3(enc2)  # [bs, 64, 64, 64]
        enc4 = self.Encoder_block4(enc3)  # [bs, 80, 32, 32]

        # Decoder modify
        dec1 = self.Decoder_block1(enc4)  # [bs, 64, 64, 64]
        dec2 = self.Decoder_block2(torch.cat([dec1, enc3], dim=1))  # [bs, 48, 128, 128]
        dec3 = self.Decoder_block3(torch.cat([dec2, enc2], dim=1))  # [bs, 32, 256, 256]
        dec4 = self.Decoder_block4(torch.cat([dec3, enc1], dim=1))  # [bs, 32, 256, 256]

        fuse1 = self.up_sample1(dec1)  # [bs, 64, 256, 256]
        fuse2 = self.up_sample2(dec2)  # [bs, 48, 256, 256]
        fuse = torch.cat([fuse1, fuse2, dec3, dec4], dim=1)  # [bs, 32+48+64+80, 256, 256]
        out_2D = self.fuse_block(fuse)  # [bs, 32, 256, 256]

        # out_2D = out_2D + x

        out_3D = self.seg_head(out_2D)
        # Take back to [W, H, D] axis order
        out_3D = out_3D.permute(0, 1, 3, 4, 2)  # [bs, C, H, W, D] -> [bs, C, W, H, D]
        return out_3D

    def pack(self, array):
        """convert a boolean array into a bitwise array."""
        array = array.reshape((-1))

        # compressing bit flags.
        # yapf: disable
        compressed = array[::8] << 7 | array[1::8] << 6  | array[2::8] << 5 | array[3::8] << 4 | array[4::8] << 3 | array[5::8] << 2 | array[6::8] << 1 | array[7::8]
        # yapf: enable

        return np.array(compressed, dtype=np.uint8)

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

        ones = torch.ones_like(target)
        target = torch.where(torch.logical_or(target == self.ignore_index, target == 0), target, ones)

        return bev_map, target

    def loss(self, batch_inputs_dict: Dict[List, Tensor], batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        voxel_dict = batch_inputs_dict["voxels"]
        bev_map, target = self.get_bev_map(voxel_dict, batch_data_samples)

        out_level_1 = self.step(bev_map.permute(0, 3, 1, 2).to(target.device))
        # calculate loss
        losses = dict()
        losses_pts = dict()
        class_weights_level_1 = self.class_weights_level_1.type_as(target)
        loss_sc_level_1 = BCE_ssc_loss(out_level_1, target, class_weights_level_1, self.alpha, self.ignore_index)
        losses_pts["loss_sc_level_1"] = loss_sc_level_1

        losses.update(losses_pts)

        return losses

    def predict(self, batch_inputs_dict, batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        voxel_dict = batch_inputs_dict["voxels"]
        bev_map, target = self.get_bev_map(voxel_dict, batch_data_samples)

        sc_pred = self.step(bev_map.permute(0, 3, 1, 2).to(target.device))
        y_pred = sc_pred.detach().cpu().numpy()  # [1, 20, 128, 128, 16]
        y_pred = np.argmax(y_pred, axis=1).astype(np.uint8)
        y_in = bev_map.detach().cpu().numpy().astype(np.uint8)
        y_pred = y_pred | y_in

        # save query proposal
        # img_path = result["img_path"]
        # frame_id = os.path.splitext(img_path[0])[0][-6:]

        # msnet3d
        # if not os.path.exists(os.path.join("./kitti/dataset/sequences_msnet3d_sweep10", img_metas[0]['sequence_id'], 'queries')):
        # os.makedirs(os.path.join("./kitti/dataset/sequences_msnet3d_sweep10", img_metas[0]['sequence_id'], 'queries'))
        # save_query_path = os.path.join("./kitti/dataset/sequences_msnet3d_sweep10", img_metas[0]['sequence_id'], 'queries', frame_id + ".query_iou5203_pre7712_rec6153")

        # y_pred_bin = self.pack(y_pred)
        # y_pred_bin.tofile(save_query_path)
        # ---------------------------------------------------------------------------------------------------

        result = dict()
        y_true = target.cpu().numpy()
        result["y_pred"] = y_pred
        result["y_true"] = y_true
        result = list([result])
        return result


class SegmentationHead(nn.Module):
    """
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    """

    def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
        super().__init__()

        # First convolution
        self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList([nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        self.conv2 = nn.ModuleList([nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
        self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
        self.relu = nn.ReLU(inplace=True)

        # Convolution for output
        self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

    def forward(self, x_in):
        # Dimension exapension
        x_in = x_in[:, None, :, :, :]

        # Convolution to go from inplanes to planes features...
        x_in = self.relu(self.conv0(x_in))

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        x_in = self.conv_classes(x_in)

        return x_in
