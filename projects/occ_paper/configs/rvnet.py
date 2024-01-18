from projects.occ_paper.occ_paper.rvnet import LMSCNet_SS
from projects.occ_paper.occ_paper.hungarian_assigner_3d import HungarianAssigner3D
from projects.occ_paper.occ_paper.match_cost import FocalLossCost, BBox3DL1Cost, IoUCost
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor

_gamma_ = 0
_alpha_ = 0.54


point_cloud_range = [0, -25.6, -2.0, 51.2, 25.6, 4.4]
voxel_size = [0.2, 0.2, 0.2]

bev_h_ = 128
bev_w_ = 128
queue_length = 3  # each sequence contains `queue_length` frames.

model = dict(
    type=LMSCNet_SS,
    class_num=2,
    input_dimensions=[256, 32, 256],
    out_scale="1_2",
    gamma=_gamma_,
    alpha=_alpha_,
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
