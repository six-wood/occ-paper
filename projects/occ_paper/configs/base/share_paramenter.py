# Description: This file contains the shared parameters for the project.
# data parameters
point_cloud_range = [0, -25.6, -2.0, 51.2, 25.6, 4.4]
fov_horizontal = [-90.0, 90.0]
fov_vertical = [-25.0, 3.0]
voxel_size = [0.2, 0.2, 0.2]
grid_size = [256, 256, 32]
scale = "1_1"
# loss parameters
ignore_index = 255
free_index = 0
number_classes = 20
# class_weight = [0.45, 0.55]
range_encoder_channel = 64
fuse_channel = 32

semantic_kitti_class_frequencies = [
    5.41773033e09,
    1.57835390e07,
    1.25136000e05,
    1.18809000e05,
    6.46799000e05,
    8.21951000e05,
    2.62978000e05,
    2.83696000e05,
    2.04750000e05,
    6.16887030e07,
    4.50296100e06,
    4.48836500e07,
    2.26992300e06,
    5.68402180e07,
    1.57196520e07,
    1.58442623e08,
    2.06162300e06,
    3.69705220e07,
    1.15198800e06,
    3.34146000e05,
]
