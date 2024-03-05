import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.structures import Det3DDataSample
from typing import Dict, List, Optional, Sequence, Union
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.utils import ConfigType, OptConfigType
from mmdet3d.structures.det3d_data_sample import SampleList
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from .utils import make_res_layer


class MSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], groups=[2, 4, 8], conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(MSBlock, self).__init__()
        self.MSC = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            self.MSC.append(
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_sizes[i],
                    stride=1,
                    padding=(kernel_sizes[i] - 1) // 2,
                    groups=groups[i],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i in range(len(self.MSC)):
            out = out + self.MSC[i](x)
        return out


class BevNet(BaseModule):
    def __init__(
        self,
        input_dimensions=32,
        stem_channels: int = 32,
        num_stages: int = 3,
        stage_blocks: Sequence[int] = (2, 4, 2),
        strides: Sequence[int] = (2, 2, 2),
        dilations: Sequence[int] = (1, 1, 1),
        encoder_out_channels: Sequence[int] = (48, 64, 80),
        decoder_out_channels: Sequence[int] = (64, 48, 32),
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN"),
        act_cfg: ConfigType = dict(type="LeakyReLU"),
    ):
        super().__init__()
        """
        SSCNet architecture
        :param N: number of classes to be predicted (i.e. 12 for NYUv2)
        """
        assert len(encoder_out_channels) == len(decoder_out_channels) == num_stages, (
            "The length of encoder_out_channels, decoder_out_channels " "should be equal to num_stages"
        )
        # input downsample

        # model parameters
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        in_channel, stem_out = input_dimensions, stem_channels
        self.stem = self._make_conv_layer(in_channel, stem_out)
        # encoder_channels
        inplanes = stem_out
        e_out2 = encoder_out_channels[0]
        e_out3 = encoder_out_channels[1]
        e_out4 = encoder_out_channels[2]

        # encode block
        self.res_layers = []
        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = encoder_out_channels[i]
            res_layer = make_res_layer(
                inplanes=inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            inplanes = planes
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        # decoder channels
        d_in1 = e_out4
        d_out1 = decoder_out_channels[0]
        d_out2 = decoder_out_channels[1]
        d_out3 = decoder_out_channels[2]

        # decode block 1
        self.up_sample1 = nn.ConvTranspose2d(d_in1, d_in1, kernel_size=2, padding=0, stride=2, bias=False)
        self.conv_layer1 = self._make_conv_layer(d_in1 + e_out3, d_out1)
        self.MSB1 = MSBlock(d_out1, d_out1, kernel_sizes=[1, 3, 5], groups=[4, 8, 16], conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # decode block 2
        self.up_sample2 = nn.ConvTranspose2d(d_out1, d_out1, kernel_size=2, padding=0, stride=2, bias=False)
        self.up_sample2_1 = nn.ConvTranspose2d(d_in1, d_in1, kernel_size=4, padding=0, stride=4, bias=False)
        self.conv_layer2 = self._make_conv_layer(d_out1 + d_in1 + e_out2, d_out2)
        self.MSB2 = MSBlock(d_out2, d_out2, kernel_sizes=[3, 5, 7], groups=[3, 6, 12], conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # decode block 3
        self.up_sample3 = nn.ConvTranspose2d(d_out2, d_out2, kernel_size=2, padding=0, stride=2, bias=False)
        self.up_sample3_1 = nn.ConvTranspose2d(d_out1, d_out1, kernel_size=4, padding=0, stride=4, bias=False)
        self.up_sample3_2 = nn.ConvTranspose2d(d_in1, d_in1, kernel_size=8, padding=0, stride=8, bias=False)
        self.conv_layer3 = self._make_conv_layer(d_out2 + d_out1 + d_in1 + stem_out, d_out3)
        self.MSB3 = MSBlock(d_out3, d_out3, kernel_sizes=[5, 7, 9], groups=[2, 4, 8], conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def _make_conv_layer(self, in_channels: int, out_channels: int) -> None:  # two conv blocks in beginning
        return nn.Sequential(
            ConvModule(in_channels, out_channels, kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
            ConvModule(out_channels, out_channels, kernel_size=3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
        )

    def forward(self, bev_map: torch.Tensor = None):
        # Encoder
        x = self.stem(bev_map)  # [bs, 32, 256, 256]
        outs = [x]
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            outs.append(x)

        # Decoder1 out_1/4
        dec1 = self.up_sample1(outs[-1])  # [bs, 80, 64, 64]
        dec1 = torch.cat([dec1, outs[-2]], dim=1)  # [bs, 80+64, 64, 64]
        dec1 = self.conv_layer1(dec1)  # [bs, 64, 64, 64]
        dec1 = self.MSB1(dec1)

        # Decoder2 out_1/2
        dec2 = self.up_sample2(dec1)  # [bs, 64, 128, 128]
        fuse2_1 = self.up_sample2_1(outs[-1])  # [bs, 80, 128, 128]
        dec2 = torch.cat([dec2, outs[-3], fuse2_1], dim=1)  # [bs, 64+48+80, 128, 128]
        dec2 = self.conv_layer2(dec2)  # [bs, 48, 128, 128]
        dec2 = self.MSB2(dec2)

        # Decoder3 out_1
        dec3 = self.up_sample3(dec2)  # [bs, 48, 256, 256]
        fuse3_1 = self.up_sample3_1(dec1)  # [bs, 64, 256, 256]
        fuse3_2 = self.up_sample3_2(outs[-1])  # [bs, 80, 256, 256]
        dec3 = torch.cat([dec3, outs[-4], fuse3_1, fuse3_2], dim=1)  # [bs, 48+32+64+80, 256, 256]
        dec3 = self.conv_layer3(dec3)  # [bs, 32, 256, 256]
        dec3 = self.MSB3(dec3)
        # dec3 = self.atten_block3(dec3, bev_map)

        return dec3


class ScHead(nn.Module):
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

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
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


@MODELS.register_module()
class ScNet(MVXTwoStageDetector):
    def __init__(
        self,
        free_index: int = 0,
        ignore_index: int = 255,
        test_save_dir: str = None,
        loss_sc: ConfigType = dict(
            type=CrossEntropyLoss,
            class_weight=[0.446, 0.505],
            loss_weight=1.0,
        ),
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="LeakyReLU"),
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
        **kwargs,
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
        self.grid_shape = self.data_preprocessor.voxel_layer.grid_shape
        self.pc_range = self.data_preprocessor.voxel_layer.point_cloud_range
        self.voxel_size = self.data_preprocessor.voxel_layer.voxel_size
        self.free_index = free_index
        self.ignore_index = ignore_index
        self.test_save_dir = test_save_dir

        self.backbone = BevNet(
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.sc_head = ScHead(1, 8, 2, [1, 2, 3])
        self.loss_sc = MODELS.build(loss_sc)

        np.set_printoptions(threshold=np.inf, linewidth=np.inf, formatter={"int": "{:7d}".format})

    def extract_pts_feat(self, voxel_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
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
        bev_feature = self.backbone(bev_map)  # channel first
        return bev_feature

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> torch.Tensor:
        """Concat voxel-wise Groud Truth."""
        gt_semantic_segs = [data_sample.gt_pts_seg.voxel_label.long() for data_sample in batch_data_samples]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss(self, batch_inputs_dict: Dict[List, torch.Tensor], batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        voxel_dict = batch_inputs_dict.get("voxels", None)
        bev_fea = self.extract_pts_feat(voxel_dict)
        sc_logits = self.sc_head(bev_fea)
        sc_logits = torch.permute(sc_logits, (0, 1, 3, 4, 2)).contiguous()  # B, C, Z, H, W -> B, C, H, W, Z
        seg_label = self._stack_batch_gt(batch_data_samples)
        geo_label = torch.where(torch.logical_and(seg_label != self.free_index, seg_label != self.ignore_index), 1, seg_label)
        loss = dict()
        loss["loss_geo"] = self.loss_sc(sc_logits, geo_label, ignore_index=self.ignore_index)
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples: List[Det3DDataSample], **kwargs) -> List[Det3DDataSample]:
        voxel_dict = batch_inputs_dict.get("voxels", None)
        bev_fea = self.extract_pts_feat(voxel_dict)
        sc_logits = self.sc_head(bev_fea)
        sc_logits = torch.permute(sc_logits, (0, 1, 3, 4, 2)).contiguous()
        results = self.postprocess_result(sc_logits, batch_data_samples)
        return results

    def postprocess_result(self, sc_logits: torch.Tensor, batch_data_samples):
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

        sc_gt = [data_sample.gt_pts_seg.voxel_label.long() for data_sample in batch_data_samples]
        sc_gt = torch.stack(sc_gt, dim=0)
        sc_gt = torch.where(torch.logical_and(sc_gt != self.free_index, sc_gt != self.ignore_index), 1, sc_gt).cpu().numpy()

        sc_pred = sc_logits.argmax(dim=1).cpu().numpy()

        result = dict()
        result["y_pred"] = sc_pred
        result["y_true"] = sc_gt
        result = list([result])
        return result

    def occupied_voxels_to_pc(self, occupancy_grid, occupancy_label):
        """Saves only occupied voxels to a text file."""
        # Reshape the grid
        grid_shape = self.grid_shape
        x_scale, y_scale, z_scale = self.voxel_size
        x_offset, y_offset, z_offset = self.pc_range[:3]

        reshaped_grid = occupancy_grid.reshape(grid_shape)
        reshaped_label = occupancy_label.reshape(grid_shape)

        # Generate grid coordinates
        x, y, z = np.meshgrid(np.arange(grid_shape[0]), np.arange(grid_shape[1]), np.arange(grid_shape[2]), indexing="ij")
        x = x * x_scale + x_offset
        y = y * y_scale + y_offset
        z = z * z_scale + z_offset

        # Flatten and filter out unoccupied voxels
        coordinates = np.vstack((x.ravel(), y.ravel(), z.ravel(), reshaped_grid.ravel(), reshaped_label.ravel())).T
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

        data = self.data_preprocessor(data, False)
        batch_inputs_dict = data["inputs"]
        batch_data_samples = data["data_samples"]
        voxel_dict = batch_inputs_dict["voxels"]

        seg_label = self._stack_batch_gt(batch_data_samples).cpu().numpy()
        bev_fea = self.extract_pts_feat(voxel_dict)
        sc_logits = self.sc_head(bev_fea)
        sc_logits = torch.permute(sc_logits, (0, 1, 3, 4, 2)).contiguous()
        seg_pred = sc_logits.argmax(dim=1).cpu().numpy()

        for i, data_sample in enumerate(batch_data_samples):
            metainfo = data_sample.get("metainfo", {})
            lidar_path = metainfo.get("lidar_path", None)
            seq = lidar_path.split("/")[-3]
            name = lidar_path.split("/")[-1]
            sc_save_path = f"{self.test_save_dir}/{seq}/{'velodyne'}/{name}"
            txt_save_path = f"{self.test_save_dir}/{seq}/{'txt'}/{name}".replace(".bin", ".txt")
            label_save_path = f"{self.test_save_dir}/{seq}/{'labels'}/{name}".replace(".bin", ".label")
            pred_pc = self.occupied_voxels_to_pc(seg_pred[i, :], seg_label[i, :])
            pred_geo = pred_pc[:, :3].astype(np.float32)
            pred_sem = pred_pc[:, 4].astype(np.int32)

            pred_txt = np.hstack((pred_geo, pred_sem[:, None]))
            np.savetxt(txt_save_path, pred_txt, fmt="%f %f %f %f")

            # label frequency counr
            sample_label_count = np.bincount(pred_sem)
            all_label_count = np.bincount(seg_label[i, :].ravel())
            index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 255]
            print(f"Sample label count: {sample_label_count[index]}")
            print(f"All label count   : {all_label_count[index]}")
            print("====================================")

            pred_geo.tofile(sc_save_path)
            pred_sem.tofile(label_save_path)

            data_sample.set_data(
                {
                    "y_true": seg_label[i, :],
                    "y_pred": seg_pred[i, :],
                    "pred_pts_geo": pred_geo,
                    "pred_pts_seg": PointData(**{"pts_semantic_mask": pred_sem}),
                }
            )
        return batch_data_samples
