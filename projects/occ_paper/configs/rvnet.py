from projects.occ_paper.occ_paper.rvnet_backbone import LMSCNet_SS
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.models.voxel_encoders.pillar_encoder import DynamicPillarFeatureNet
from mmengine.config import read_base
import torch.nn as nn

with read_base():
    from .share_paramenter import *

_gamma_ = 0
_alpha_ = 0.54

model = dict(
    type=LMSCNet_SS,
    class_num=2,
    input_dimensions=input_demiensions,
    out_scale=scale,
    gamma=_gamma_,
    alpha=_alpha_,
    ignore_index=ignore_index,
    # model training and testing settings
    data_preprocessor=dict(
        type=Det3DDataPreprocessor,
        voxel=True,
        voxel_type="dynamic",
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1),
        ),
    ),
)
