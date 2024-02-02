import numpy as np

# 体素网格的大小
voxel_grid_size = np.array([256, 256, 32])

# 空间范围
space_range = np.array([[0, 51.2], [-25.6, 25.6], [-2.0, 4.4]])

# fov
scale = np.pi / 180
fov_up = 3.0 * scale
fov_down = -25.0 * scale
fov_left = -90.0 * scale
fov_right = 90.0 * scale

# 计算每个轴向上的体素尺寸
voxel_size = (space_range[:, 1] - space_range[:, 0]) / voxel_grid_size

# 使用numpy内置函数来计算体素中心坐标

# 创建每个维度上的坐标网格
x = np.linspace(space_range[0, 0] + voxel_size[0] / 2, space_range[0, 1] - voxel_size[0] / 2, voxel_grid_size[0])
y = np.linspace(space_range[1, 0] + voxel_size[1] / 2, space_range[1, 1] - voxel_size[1] / 2, voxel_grid_size[1])
z = np.linspace(space_range[2, 0] + voxel_size[2] / 2, space_range[2, 1] - voxel_size[2] / 2, voxel_grid_size[2])

# 使用meshgrid生成三维网格
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# 将三维网格堆叠成体素中心坐标
voxel_centers_np = np.stack((X, Y, Z), axis=-1)

# 显示一些体素中心的坐标作为例子
voxel_centers_np[0, 0, 0], voxel_centers_np[-1, -1, -1]  # 显示第一个和最后一个体素的中心坐标

# 从voxel_centers_np提取X, Y, Z坐标
X_voxel = voxel_centers_np[..., 0]
Y_voxel = voxel_centers_np[..., 1]
Z_voxel = voxel_centers_np[..., 2]

# 计算半径 r
r_voxel = np.sqrt(X_voxel**2 + Y_voxel**2 + Z_voxel**2)

# 计算方位角 phi
yaw = np.arctan2(Y_voxel, X_voxel)

# 计算俯角
pitch = np.arcsin(Z_voxel / r_voxel)

# 将极坐标结果堆叠成一个数组
polar_coordinates_voxel = np.stack((r_voxel, pitch, yaw), axis=-1)

# 显示一些体素的极坐标作为例子
polar_coordinates_voxel[0, 0, 0], polar_coordinates_voxel[-1, -1, -1]  # 显示第一个和最后一个体素的极坐标

vertical_fov = (-3, 25)  # 垂直视场角范围（度）
horizontal_fov = (-90, 90)  # 水平视场角范围（度）

# 转换为弧度
vertical_fov_rad = np.radians(vertical_fov)
horizontal_fov_rad = np.radians(horizontal_fov)

# 定义分辨率参数
vertical_resolution = 64
horizontal_resolution = 1024

# 过滤视场范围内的点
mask = (pitch >= vertical_fov_rad[0]) & (pitch <= vertical_fov_rad[1]) & (yaw >= horizontal_fov_rad[0]) & (yaw <= horizontal_fov_rad[1])

# 计算映射后的像素坐标
vertical_pixel_coord = (pitch[mask] - vertical_fov_rad[0]) / (vertical_fov_rad[1] - vertical_fov_rad[0]) * vertical_resolution
horizontal_pixel_coord = (yaw[mask] - horizontal_fov_rad[0]) / (horizontal_fov_rad[1] - horizontal_fov_rad[0]) * horizontal_resolution

# 将坐标舍入为整数像素索引
vertical_pixel_coord = np.round(vertical_pixel_coord).astype(int)
horizontal_pixel_coord = np.round(horizontal_pixel_coord).astype(int)

# 显示一些计算得到的像素坐标作为例子
piexl_coord = np.stack((vertical_pixel_coord, horizontal_pixel_coord), axis=-1)

unique_pixel_coord = np.unique(piexl_coord, axis=0)
piexl_coord[0], piexl_coord[-1]  # 显示第一个和最后一个像素坐标

all_size = piexl_coord.shape[0]
unique_size = unique_pixel_coord.shape[0]
print(all_size, unique_size, unique_size / all_size)
print(64 * 1024, unique_size, unique_size / (64 * 1024))
