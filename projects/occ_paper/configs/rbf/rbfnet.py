import torch.nn as nn
from projects.occ_paper.occ_paper.models.rbfnet import RBFNet
from projects.occ_paper.occ_paper.models.ssc_net import SscNet
from projects.occ_paper.occ_paper.models.ssc_head import SscHead
from projects.occ_paper.occ_paper.models.ssc_loss import BECLoss
from projects.occ_paper.occ_paper.data_preprocessor import SccDataPreprocessor
from mmengine.config import read_base

with read_base():
    from .share_paramenter import *

model = dict(
    type=SscNet,
    data_preprocessor=dict(
        type=SccDataPreprocessor,
        voxel=True,
        voxel_type="dynamic",
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1),
        ),
    ),
    pts_backbone=dict(
        type=RBFNet,
    ),
    pts_scc_head=dict(
        type=SscHead,
        inplanes=1,
        planes=8,
        nbr_classes=number_classes,
        dilations_conv_list=[1, 2, 3],
    ),
    pts_scc_loss=dict(
        type=BECLoss,
    ),
)
