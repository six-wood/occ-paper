"""
Code partly taken from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/data/labels_downscale.py
"""
import numpy as np
from tqdm import tqdm
import os
import glob
import io_data as SemanticKittiIO
import argparse

# from numba import jit, njit


# @jit(nopython=False, parallel=True, fastmath=True, cache=True)
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
    grid_size = ((pc_range[3:] - pc_range[:3]) / voxel_size).astype(np.int32)

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


# @jit(nopython=False, parallel=True, fastmath=True, cache=True)
def compute_valid_mask(view_grid, center, free_id=0, ignore_id=255):
    view_grid_ = view_grid.copy()
    # view_grid_[view_grid_ == ignore_id] = free_id
    voxels_mask = np.zeros_like(view_grid_, dtype=bool)
    x_voxels, y_voxels, z_voxels = view_grid_.shape
    x_center, y_center, z_center = center

    # Precompute t_values
    # t_values = np.linspace(0, 1, 101).reshape(-1, 1)  # Including the endpoint
    # t_values = np.linspace(0, 1, 100).reshape(-1, 1)  # Including the endpoint

    for x, y, z in np.ndindex(view_grid_.shape):
        if view_grid_[x, y, z] != free_id:
            z_step = np.abs(z - z_center).astype(np.int32)
            t_values = np.linspace(0, 1, z_step + 1).reshape(-1, 1)
            # Vectorized computation of indices
            indices = (np.array([x, y, z]) + t_values * (np.array([x_center, y_center, z_center]) - np.array([x, y, z]))).astype(np.int32)

            # Remove duplicate index combinations

            valid_id = np.unique(indices, axis=0)

            if (view_grid_[valid_id[:, 0], valid_id[:, 1], valid_id[:, 2]] != free_id).sum() > 1.0:
                continue

            voxels_mask[valid_id[:, 0], valid_id[:, 1], valid_id[:, 2]] = True

    return voxels_mask


# @jit(nopython=False, parallel=True, fastmath=True, cache=True)
def label_rectification(grid_ind, voxel_label, instance_label, dynamic_classes=[4, 5, 6, 7, 8], voxel_shape=(256, 256, 32), ignore_class_label=255):
    segmentation_label = voxel_label[grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2]]

    for c in dynamic_classes:
        voxel_pos_class_c = voxel_label == c
        instance_label_class_c = instance_label[segmentation_label == c].squeeze(1)

        if len(instance_label_class_c) == 0:
            pos_to_remove = voxel_pos_class_c

        elif len(instance_label_class_c) > 0 and np.sum(voxel_pos_class_c) > 0:
            mask_class_c = np.zeros(voxel_shape, dtype=bool)
            point_pos_class_c = grid_ind[segmentation_label == c]
            uniq_instance_label_class_c = np.unique(instance_label_class_c)

            for i in uniq_instance_label_class_c:
                point_pos_instance_i = point_pos_class_c[instance_label_class_c == i]
                x_max, y_max, z_max = np.amax(point_pos_instance_i, axis=0)
                x_min, y_min, z_min = np.amin(point_pos_instance_i, axis=0)

                mask_class_c[x_min:x_max, y_min:y_max, z_min:z_max] = True

            pos_to_remove = voxel_pos_class_c & ~mask_class_c

        voxel_label[pos_to_remove] = ignore_class_label

    return voxel_label


sweep = 10


def main(config):
    scene_size = (256, 256, 32)
    sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    remap_lut = SemanticKittiIO._get_remap_lut(
        os.path.join(
            "./semantic-kitti.yaml",
        )
    )
    downscaling = {"1_1": 1}
    min_bound = np.array([0, -25.6, -2])
    max_bound = np.array([51.2, 25.6, 4.4])
    intervals = np.array([0.2, 0.2, 0.2])
    # INVISIBILITY = ~compute_visibility_mask()
    # grid_center = (np.array([0, 0, 0]) - min_bound) / intervals
    for sequence in sequences:
        sequence_path = os.path.join(config.kitti_root, "dataset", "sequences", sequence)
        pc_paths = sorted(glob.glob(os.path.join(sequence_path, "velodyne", "*.bin")))
        pc_labels = sorted(glob.glob(os.path.join(sequence_path, "labels", "*.label")))
        label_paths = sorted(glob.glob(os.path.join(sequence_path, "voxels", "*.label")))
        invalid_paths = sorted(glob.glob(os.path.join(sequence_path, "voxels", "*.invalid")))
        out_dir = os.path.join(config.kitti_preprocess_root, "labels", sequence)
        os.makedirs(out_dir, exist_ok=True)

        # downscaling = {"1_1": 1, "1_2": 2}
        if sequence != "08":
            print("use yourself valid mask")
        for i in tqdm(range(0, len(label_paths), sweep)):
            frame_id, extension = os.path.splitext(os.path.basename(label_paths[i]))

            PC = SemanticKittiIO._read_pointcloud_SemKITTI(pc_paths[i])[:, :3]
            PC_INSTANCE = np.fromfile(pc_labels[i], dtype=np.uint32).reshape(-1, 1) & 0xFFFF  # 0xFFFF is the mask to get the lower 16 bits

            box_filter = np.logical_and(
                np.logical_and(PC[:, 0] >= min_bound[0], PC[:, 0] < max_bound[0]),
                np.logical_and(PC[:, 1] >= min_bound[1], PC[:, 1] < max_bound[1]),
                np.logical_and(PC[:, 2] >= min_bound[2], PC[:, 2] < max_bound[2]),
            )
            PC = PC[box_filter]
            PC_INSTANCE = PC_INSTANCE[box_filter]
            PC_INSTANCE = remap_lut[PC_INSTANCE.astype(np.uint16)].astype(np.float32)
            grid_ind = (np.floor((np.clip(PC, min_bound, max_bound) - min_bound) / intervals)).astype(np.int32)
            LABEL = SemanticKittiIO._read_label_SemKITTI(label_paths[i])
            INVALID = SemanticKittiIO._read_invalid_SemKITTI(invalid_paths[i])
            INVALID = np.isclose(INVALID, 1)
            LABEL = remap_lut[LABEL.astype(np.uint16)].astype(np.float32)  # Remap 20 classes semanticKITTI SSC
            # Setting to unknown all voxels marked on invalid mask...
            LABEL = LABEL.reshape([256, 256, 32])
            INVALID = INVALID.reshape([256, 256, 32])

            for scale in downscaling:
                filename = frame_id + "_" + scale + ".npy"
                label_filename = os.path.join(out_dir, filename)
                # If files have not been created...
                # if not os.path.exists(label_filename):
                # if scale != "1_1":
                #     LABEL_ds = _downsample_label(LABEL, (256, 256, 32), downscaling[scale])
                # else:
                #     LABEL_ds = LABEL
                # LABEL_ds = label_rectification(grid_ind, LABEL_ds, PC_INSTANCE)WWW
                # np.save(label_filename, LABEL_ds)
                # print("wrote to", label_filename)
                LABEL_ds = LABEL
                LABEL_ds = label_rectification(grid_ind, LABEL_ds, PC_INSTANCE)

                # if sequence != "08":
                #     MYVALID = compute_valid_mask(LABEL_ds, grid_center, 0, 255)
                #     LABEL_ds[~MYVALID] = 255
                #     LABEL_ds[INVISIBILITY] = 255

                LABEL[INVALID] = 255

                np.save(label_filename, LABEL_ds)
                # print("wrote to", label_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("./label_preprocess.py")
    parser.add_argument(
        "--kitti_root",
        "-r",
        type=str,
        required=True,
        help="kitti_root",
    )

    parser.add_argument(
        "--kitti_preprocess_root",
        "-p",
        type=str,
        required=True,
        help="kitti_preprocess_root",
    )
    config, unparsed = parser.parse_known_args()
    main(config)
