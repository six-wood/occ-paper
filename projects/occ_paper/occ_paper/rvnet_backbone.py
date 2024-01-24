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
from .ssc_loss import BCE_ssc_loss


@MODELS.register_module()
class LMSCNet_SS(MVXTwoStageDetector):
    def __init__(
        self,
        class_num=None,
        input_dimensions=None,
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
        self.out_scale = out_scale
        self.nbr_classes = class_num
        self.gamma = gamma
        self.alpha = alpha
        self.input_dimensions = f = input_dimensions  # Grid dimensions should be (W, H, D).. z or height being axis 1
        self.ignore_index = ignore_index

        self.pool = nn.MaxPool2d(2)  # [F=2; S=2; P=0; D=1]

        self.pooling = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.class_frequencies_level1 = np.array([5.41773033e09, 4.03113667e08])

        self.class_weights_level_1 = torch.from_numpy(1 / np.log(self.class_frequencies_level1 + 0.001))

        self.Encoder_block1 = nn.Sequential(
            nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )

        self.Encoder_block2 = self._make_encoder_block(f, int(f * 1.5))
        self.Encoder_block3 = self._make_encoder_block(int(f * 1.5), int(f * 2))
        self.Encoder_block4 = self._make_encoder_block(int(f * 2), int(f * 2.5))

        # Treatment output 1:8
        self.conv_out_scale_1_8 = nn.Conv2d(int(f * 2.5), int(f / 8), kernel_size=3, padding=1, stride=1)  # channel->H/8
        self.deconv_1_8__1_2 = nn.ConvTranspose2d(int(f / 8), int(f / 8), kernel_size=4, padding=0, stride=4)  # 0.125->0.5
        self.deconv_1_8__1_1 = nn.ConvTranspose2d(int(f / 8), int(f / 8), kernel_size=8, padding=0, stride=8)  # 0.125->1

        # Treatment output 1:4
        if self.out_scale == "1_4" or self.out_scale == "1_2" or self.out_scale == "1_1":
            self.deconv1_8 = nn.ConvTranspose2d(int(f / 8), int(f / 8), kernel_size=6, padding=2, stride=2)  # 0.125->0.25
            self.conv1_4 = nn.Conv2d(
                int(f * 2) + int(f / 8), int(f * 2), kernel_size=3, padding=1, stride=1
            )  # fuse feature maps from encoder block 3 and deconv1_8
            self.conv_out_scale_1_4 = nn.Conv2d(int(f * 2), int(f / 4), kernel_size=3, padding=1, stride=1)  # channel->H/4
            self.deconv_1_4__1_1 = nn.ConvTranspose2d(int(f / 4), int(f / 4), kernel_size=4, padding=0, stride=4)  # 0.25->1

        # Treatment output 1:2
        if self.out_scale == "1_2" or self.out_scale == "1_1":
            self.deconv1_4 = nn.ConvTranspose2d(int(f / 4), int(f / 4), kernel_size=6, padding=2, stride=2)  # 0.25->0.5
            self.conv1_2 = nn.Conv2d(
                int(f * 1.5) + int(f / 4) + int(f / 8),
                int(f * 1.5),
                kernel_size=3,
                padding=1,
                stride=1,
            )  # fuse feature maps from encoder block 2, deconv1_4 and deconv1_8
            self.conv_out_scale_1_2 = nn.Conv2d(int(f * 1.5), int(f / 2), kernel_size=3, padding=1, stride=1)  # channel->H/2

        # Treatment output 1:1
        if self.out_scale == "1_1":
            self.deconv1_2 = nn.ConvTranspose2d(int(f / 2), int(f / 2), kernel_size=6, padding=2, stride=2)  # 0.5->1
            self.conv1_1 = nn.Conv2d(
                int(f / 8) + int(f / 4) + int(f / 2) + int(f),
                f,
                kernel_size=3,
                padding=1,
                stride=1,
            )  # fuse feature maps from encoder block 1, deconv1_2, deconv1_4 and deconv1_8

        if self.out_scale == "1_1":
            self.seg_head_1_1 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
        elif self.out_scale == "1_2":
            self.seg_head_1_2 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
        elif self.out_scale == "1_4":
            self.seg_head_1_4 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
        elif self.out_scale == "1_8":
            self.seg_head_1_8 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])

    def _make_encoder_block(self, inplanes, planes):
        return nn.Sequential(
            nn.MaxPool2d(2),  # 1->0.5
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )

    def step(self, input):
        # input = x['3D_OCCUPANCY']  # Input to LMSCNet model is 3D occupancy big scale (1:1) [bs, 1, W, H, D]
        # input = torch.squeeze(input, dim=1).permute(0, 2, 1, 3)  # Reshaping to the right way for 2D convs [bs, H, W, D]

        # print(input.shape) [4, 32, 256, 256]

        # Encoder block
        _skip_1_1 = self.Encoder_block1(input)
        # print('_skip_1_1.shape', _skip_1_1.shape)  # [1, 32, 256, 256]
        _skip_1_2 = self.Encoder_block2(_skip_1_1)
        # print('_skip_1_2.shape', _skip_1_2.shape)  # [1, 48, 128, 128]
        _skip_1_4 = self.Encoder_block3(_skip_1_2)
        # print('_skip_1_4.shape', _skip_1_4.shape)  # [1, 64, 64, 64]
        _skip_1_8 = self.Encoder_block4(_skip_1_4)
        # print('_skip_1_8.shape', _skip_1_8.shape)  # [1, 80, 32, 32]

        # Out 1_8
        out_scale_1_8__2D = self.conv_out_scale_1_8(_skip_1_8)  # [1, 4, 32, 32]
        # if self.out_scale == "1_2":
        #     # Out 1_4
        #     out = self.deconv1_8(out_scale_1_8__2D)  # [1, 4, 64, 64]
        #     out = torch.cat((out, _skip_1_4), 1)  # [1, 68, 64, 64]
        #     out = F.relu(self.conv1_4(out))  # [1, 68, 64, 64]
        #     out_scale_1_4__2D = self.conv_out_scale_1_4(out)  # [1, 8, 64, 64]

        #     # Out 1_2
        #     out = self.deconv1_4(out_scale_1_4__2D)  # [1, 8, 128, 128]
        #     out = torch.cat((out, _skip_1_2, self.deconv_1_8__1_2(out_scale_1_8__2D)), 1)  # [1, 60, 128, 128]
        #     out = F.relu(self.conv1_2(out))  # torch.Size([1, 60, 128, 128])
        #     out_scale_1_2__2D = self.conv_out_scale_1_2(out)  # torch.Size([1, 16, 128, 128])

        #     out_scale_1_2__3D = self.seg_head_1_2(out_scale_1_2__2D)  # [1, 20, 16, 128, 128]
        #     out_scale_1_2__3D = out_scale_1_2__3D.permute(0, 1, 3, 4, 2)  # [1, 20, 128, 128, 16]
        #     return out_scale_1_2__3D

        # if self.out_scale == "1_1":
        # basic:up_sample->cat->conv->relu->conv
        # Out 1_4
        out = self.deconv1_8(out_scale_1_8__2D)  # [1, 4, 64, 64]
        # print("out.shape", out.shape)  # [1, 4, 64, 64]
        out = torch.cat((out, _skip_1_4), 1)  # [1, 68, 64, 64]
        out = F.relu(self.conv1_4(out))  # [1, 68, 64, 64]
        out_scale_1_4__2D = self.conv_out_scale_1_4(out)  # [1, 8, 64, 64]
        # print('out_scale_1_4__2D.shape', out_scale_1_4__2D.shape)  # [1, 8, 64, 64]

        # Out 1_2
        out = self.deconv1_4(out_scale_1_4__2D)  # [1, 8, 128, 128]
        # print("out.shape", out.shape)  # [1, 8, 128, 128]
        out = torch.cat((out, _skip_1_2, self.deconv_1_8__1_2(out_scale_1_8__2D)), 1)  # [1, 48+8+4, 128, 128]
        out = F.relu(self.conv1_2(out))  # torch.Size([1, 60, 128, 128])
        out_scale_1_2__2D = self.conv_out_scale_1_2(out)  # torch.Size([1, 16, 128, 128])
        # print('out_scale_1_2__2D.shape', out_scale_1_2__2D.shape)  # [1, 16, 128, 128]

        # Out 1_1
        out = self.deconv1_2(out_scale_1_2__2D)  # [1, 16, 256, 256]
        out = torch.cat(
            (
                out,
                _skip_1_1,
                self.deconv_1_4__1_1(out_scale_1_4__2D),
                self.deconv_1_8__1_1(out_scale_1_8__2D),
            ),
            1,
        )  # [1, 32+16+8+4, 256, 256]
        out_scale_1_1__2D = F.relu(self.conv1_1(out) + input)  # [bs, 32, 256, 256]

        out_scale_1_1__3D = self.seg_head_1_1(out_scale_1_1__2D)
        # Take back to [W, H, D] axis order
        out_scale_1_1__3D = out_scale_1_1__3D.permute(0, 1, 3, 4, 2)  # [bs, C, H, W, D] -> [bs, C, W, H, D]
        return out_scale_1_1__3D

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
        y_pred = np.argmax(y_pred, axis=1).astype(np.uint8)  # [1, 128, 128, 16]

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
