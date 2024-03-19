import os
import yaml
import torch
import chamfer
import open3d as o3d
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from mmcv.ops.points_in_boxes import points_in_boxes_cpu
from scipy.spatial.transform import Rotation
from copy import deepcopy


free_index = 35

class_mapping = {
    0: 0,  # "void"
    1: 0,  # "dirt"
    3: 1,  # "grass"
    4: 2,  # "tree"
    5: 3,  # "pole"
    6: 4,  # "water"
    7: 0,  # "sky"
    8: 5,  # "vehicle"
    9: 0,  # "object"
    10: 0,  # "asphalt"
    12: 0,  # "building"
    15: 6,  # "log"
    17: 7,  # "person"
    18: 8,  # "fence"
    19: 9,  # "bush"
    23: 10,  # "concrete"
    27: 11,  # "barrier"
    31: 12,  # "puddle"
    33: 13,  # "mud"
    34: 14,  # "rubble"
    free_index: 15,  # "free"
}

class_map_array = np.zeros(36)
for key in class_mapping.keys():
    class_map_array[key] = class_mapping[key]
class_map_array = class_map_array.astype(np.int32)


def voxel2points(visible_voxel, semantic_voxel):
    # convert voxel to points
    visible_voxel = np.array(visible_voxel)
    points = np.argwhere(visible_voxel > 0)
    points = np.concatenate(
        [points, semantic_voxel[visible_voxel > 0][:, np.newaxis]], axis=1
    )
    return points


def voxel2points_without_semantic(visible_voxel):
    # convert voxel to points
    visible_voxel = np.array(visible_voxel)
    points = np.argwhere(visible_voxel > 0)
    return points


def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates to spherical (r, theta, phi).
    r: Radius (distance to origin)
    theta: Azimuthal angle (angle in the xy-plane from the x-axis)
    phi: Polar angle (angle from the z-axis)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return r, theta, phi


def get_visible_voxels_vector(view_grid, center, free_id=free_index):
    voxels_mask = np.zeros_like(view_grid, dtype=bool)
    x_voxels, y_voxels, z_voxels = view_grid.shape
    x_center, y_center, z_center = center

    # Precompute t_values
    # t_values = np.linspace(0, 1, 101).reshape(-1, 1)  # Including the endpoint
    t_values = np.linspace(0, 1, 100).reshape(-1, 1)  # Including the endpoint

    for i, j, k in np.ndindex(view_grid.shape):
        if view_grid[i, j, k] != free_id:
            # Vectorized computation of indices
            indices = np.round(
                np.array([i, j, k])
                + t_values
                * (np.array([x_center, y_center, z_center]) - np.array([i, j, k]))
            ).astype(int)

            # Check if the indices are within the grid boundaries
            valid_id = (
                (indices[:, 0] >= 0)
                & (indices[:, 0] < x_voxels)
                & (indices[:, 1] >= 0)
                & (indices[:, 1] < y_voxels)
                & (indices[:, 2] >= 0)
                & (indices[:, 2] < z_voxels)
            )

            # Filter the indices based on validity
            valid_id = indices[valid_id]

            # Remove duplicate index combinations
            valid_id = np.unique(valid_id, axis=0)

            if (
                view_grid[valid_id[:, 0], valid_id[:, 1], valid_id[:, 2]] != free_id
            ).sum() > 1.0:
                continue

            voxels_mask[valid_id[:, 0], valid_id[:, 1], valid_id[:, 2]] = True

    return voxels_mask


# down_sample by open3d
def down_sample(points, voxel_size):
    points = np.concatenate((points, np.zeros((points.shape[0], 2))), axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6])
    pcd_sample = pcd.voxel_down_sample(voxel_size=voxel_size)
    points_sample = np.asarray(pcd_sample.points)
    color_sample = np.asarray(pcd_sample.colors)
    points_sample = np.concatenate((points_sample, color_sample[:, :1]), axis=1)
    return points_sample


def transform_point(point, mtx):
    point_ = np.ones((point.shape[0], 4))
    point_[:, :3] = point[:, :3]
    point_ = mtx @ point_.T
    point_ = point_.T[:, :3]
    point_copy = point.copy()
    point_copy[:, :3] = point_
    return point_copy


def load_pose(path):
    pose_info = np.loadtxt(path)
    pose_info = np.concatenate((np.zeros((1, 12), np.float32), pose_info), axis=0)
    return pose_info


def load_pc(filepath):
    point = np.fromfile(filepath, dtype=np.float32)
    point = point.reshape(-1, 4)
    return point


def save_bin(filepath, point):
    point = point.reshape(-1)
    point.astype(np.float32).tofile(filepath)


def load_label(filepath):
    label = np.fromfile(filepath, dtype=np.uint32)
    label = label.reshape(-1)
    return label


def draw_registration_result_original_color(source, target, transformation):
    source_temp = deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])


def run_poisson(pcd, depth, n_threads, min_density=None):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=8
    )

    # Post-process the mesh
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    return mesh, densities


def create_mesh_from_map(
    buffer, depth, n_threads, min_density=None, point_cloud_original=None
):
    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(buffer)
    else:
        pcd = point_cloud_original

    return run_poisson(pcd, depth, n_threads, min_density)


def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud()
    for cloud in buffer:
        pcd += cloud
    if compute_normals:
        pcd.estimate_normals()

    return pcd


def preprocess_cloud(
    pcd,
    max_nn=20,
    normals=None,
):
    cloud = deepcopy(pcd)
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params)
        cloud.orient_normals_towards_camera_location()

    return cloud


def preprocess(pcd, config):
    return preprocess_cloud(pcd, config["max_nn"], normals=True)


def nn_correspondance(verts1, verts2):
    """for each vertex in verts2 find the nearest vertex in verts1

    Args:
        nx3 np.array's
    Returns:
        ([indices], [distances])

    """
    import open3d as o3d

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances


def get_mtx_from_yaml(filepath, key):
    with open(filepath, "r") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    q = data[key]["q"]
    q = np.array([q["x"], q["y"], q["z"], q["w"]])
    t = data[key]["t"]
    t = np.array([t["x"], t["y"], t["z"]])
    R_vc = Rotation.from_quat(q)
    R_vc = R_vc.as_matrix()

    RT = np.eye(4, 4)
    RT[:3, :3] = R_vc
    RT[:3, -1] = t
    RT = np.linalg.inv(RT)
    return RT


def lidar_to_world_to_lidar(
    pc, lidar_calibrated_sensor, lidar_ego_pose, cam_calibrated_sensor, cam_ego_pose
):
    pc = LidarPointCloud(pc.T)

    pc.rotate(Rotation.from_quat(lidar_calibrated_sensor["rotation"]).as_matrix())
    pc.translate(np.array(lidar_calibrated_sensor["translation"]))

    pc.rotate(Rotation.from_quat(lidar_ego_pose["rotation"]).as_matrix())
    pc.translate(np.array(lidar_ego_pose["translation"]))

    pc.translate(-np.array(cam_ego_pose["translation"]))
    pc.rotate(Rotation.from_quat(cam_ego_pose["rotation"]).as_matrix().T)

    pc.translate(-np.array(cam_calibrated_sensor["translation"]))
    pc.rotate(Rotation.from_quat(cam_calibrated_sensor["rotation"]).as_matrix().T)

    return pc


def pc2word2pc(ego_s, ego_t):
    matrix_s = ego_s.reshape(3, 4)
    matrix_t = ego_t.reshape(3, 4)
    matrix_s_inv = np.eye(4)
    matrix_s_inv[:3, :3] = matrix_s[:, :3].T
    matrix_s_inv[:3, -1] = -matrix_s[:, :3].T @ matrix_s[:, -1]

    matrix_t_ = np.eye(4)
    matrix_t_[:3, :3] = matrix_t[:, :3]
    matrix_t_[:3, -1] = matrix_t[:, -1]

    matrix = matrix_s_inv @ matrix_t_

    return matrix


def cut_point(points, P, pad_x=0, pad_y=0, img_width=1920, img_height=1200):
    fov_x = 2 * np.arctan2(img_width, 2 * P[0, 0]) * 180 / np.pi + pad_x
    fov_y = 2 * np.arctan2(img_height, 2 * P[1, 1]) * 180 / np.pi + pad_y
    R = np.eye(4)
    p_l = np.ones(points.shape)
    p_l[:, :3] = points[:, :3].copy()
    x = p_l[:, 0]
    y = p_l[:, 1]
    z = p_l[:, 2]
    xangle = np.arctan2(x, z) * 180 / np.pi
    yangle = np.arctan2(y, z) * 180 / np.pi
    flag2 = (xangle > -fov_x / 2) & (xangle < fov_x / 2)
    flag3 = (yangle > -fov_y / 2) & (yangle < fov_y / 2)
    return flag2 & flag3


# if __name__ == '__main__':
def dense_voxel_process(save_path, sequence_range, config_path):
    # load config
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    voxel_size = config["voxel_size"]
    pc_range = config["pc_range"]
    occ_size = config["occ_size"]
    max_nn = config["max_nn"]
    fov_range = config["fov_range"]

    bbox_path = os.path.join(save_path, "bbox/")
    pc_path = os.path.join(save_path, "pc/")
    label_path = os.path.join(save_path, "pc_label/")
    pose_file = os.path.join(save_path, "poses.txt")
    transfile = os.path.join(save_path, "transforms.yaml")

    camera_info = [2813.643275, 2808.326079, 969.285772, 624.049972]
    P_cma = np.ones((3, 3))
    P_cma[0, 0] = camera_info[0]
    P_cma[1, 1] = camera_info[1]
    P_cma[2, 2] = 1
    P_cma[0, 2] = camera_info[2]
    P_cma[1, 2] = camera_info[3]

    pose_info = load_pose(pose_file)
    os2pyl_mtx = get_mtx_from_yaml(transfile, "os1_cloud_node-pylon_camera_node")

    x_min, y_min, z_min, x_max, y_max, z_max = pc_range

    center = np.array([(0 - x_min), (0 - y_min), (0 - z_min)])
    cam = np.zeros((1, 3))
    cam_pyl = transform_point(cam, os2pyl_mtx)
    center_pyl = np.array(
        [(cam_pyl[0, 0] - x_min), (cam_pyl[0, 1] - y_min), (cam_pyl[0, 2] - z_min)]
    )
    voxel_center = (np.ceil(center / voxel_size)).astype(np.int32)
    voxel_center_pyl = (np.ceil(center_pyl / voxel_size)).astype(np.int32)

    # _, _, phi = cartesian_to_spherical(x_coords, y_coords, z_coords)
    # alpha = np.pi / 2 - phi  # Convert to elevation angle
    # fov_min, fov_max = fov_range
    # fov_min_rad = np.radians(fov_min)
    # fov_max_rad = np.radians(fov_max)
    # view_angle_mask = (alpha >= fov_min_rad) & (alpha <= fov_max_rad)

    for start_id in range(sequence_range[0], sequence_range[1], max_nn):
        target_pose = pose_info[start_id, :]  # pose of the first frame in the sequence

        dict_list = []

        print("sample id %d" % start_id)

        for i in range(start_id, start_id + max_nn + 1):
            pc0 = load_pc(pc_path + str(i).zfill(6) + ".bin")
            label0 = load_label(label_path + str(i).zfill(6) + ".label")

            boxes = np.load(os.path.join(bbox_path, "bbox{}.npy".format(i)))
            object_category = np.load(
                os.path.join(bbox_path, "object_category{}.npy".format(i))
            )
            boxes_token = np.load(
                os.path.join(bbox_path, "boxes_token{}.npy".format(i))
            )

            points_in_boxes = points_in_boxes_cpu(
                torch.from_numpy(pc0[:, :3][np.newaxis, :, :]),
                torch.from_numpy(boxes[np.newaxis, :]),
            )

            object_points_list = []
            j = 0
            while j < points_in_boxes.shape[-1]:
                object_points_mask = points_in_boxes[0][:, j].bool()
                object_points = pc0[object_points_mask]
                object_points_list.append(object_points)
                j = j + 1

            # print("object points sum: ", np.concatenate(object_points_list).shape)

            moving_mask = torch.ones_like(points_in_boxes)
            points_in_boxes = torch.sum(points_in_boxes * moving_mask, dim=-1).bool()
            points_mask = ~(points_in_boxes[0])

            ############################# get point mask of the vehicle itself ##########################
            self_range = config["self_range"]
            oneself_mask = torch.from_numpy(
                (np.abs(pc0[:, 0]) > self_range[0])
                | (np.abs(pc0[:, 1]) > self_range[1])
                | (np.abs(pc0[:, 2]) > self_range[2])
            )

            ############################# get static scene segment ##########################
            points_mask = points_mask & oneself_mask
            pc = pc0[points_mask]
            label = label0[points_mask]

            filter_mask = label != 17
            pc = pc[filter_mask]
            label = label[filter_mask]

            ################## coordinate conversion to the same (first) LiDAR coordinate  ##################
            source_pose = pose_info[i, :]
            source_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc[:, :3]))
            lidar_pc = source_pcd.transform(pc2word2pc(source_pose, target_pose))

            d = {
                "object_tokens": boxes_token,
                "object_points_list": object_points_list,
                "lidar_pc": np.concatenate(
                    (np.asarray(lidar_pc.points), label.reshape(-1, 1)), axis=1
                ),
                "label": label.reshape(-1, 1),
                "lidar_ego_pose": source_pose,
                "gt_bbox_3d": boxes,
                "converted_object_category": object_category,
                "pc_file_name": i,
            }
            dict_list.append(d)

        ################## concatenate all static scene segments  ########################
        lidar_pc_list = [d["lidar_pc"] for d in dict_list]
        lidar_pc = np.concatenate(lidar_pc_list, axis=0)

        ################## concatenate all object segments (including non-key frames)  ########################
        object_token_zoo = []
        object_semantic = []
        for d in dict_list:
            for i, object_token in enumerate(d["object_tokens"]):
                if object_token not in object_token_zoo:
                    if d["object_points_list"][i].shape[0] > 0:
                        object_token_zoo.append(object_token)
                        object_semantic.append(d["converted_object_category"][i])
                    else:
                        continue

        object_points_dict = {}

        for query_object_token in object_token_zoo:
            object_points_dict[query_object_token] = []
            for d in dict_list:
                for i, object_token in enumerate(d["object_tokens"]):
                    if query_object_token == object_token:
                        object_points = d["object_points_list"][i]
                        if object_points.shape[0] > 0:
                            object_points = (
                                object_points[:, :3] - d["gt_bbox_3d"][i][:3]
                            )
                            rots = d["gt_bbox_3d"][i][6]
                            Rot = Rotation.from_euler("z", -rots, degrees=False)
                            rotated_object_points = Rot.apply(object_points)
                            object_points_dict[query_object_token].append(
                                rotated_object_points
                            )
                    else:
                        continue
            object_points_dict[query_object_token] = np.concatenate(
                object_points_dict[query_object_token], axis=0
            )

        object_points_vertice = []
        for key in object_points_dict.keys():
            point_cloud = object_points_dict[key]
            object_points_vertice.append(point_cloud[:, :3])
        # print('object finish')

        # point_cloud_original = o3d.geometry.PointCloud()
        # with_normal2 = o3d.geometry.PointCloud()
        # point_cloud_original.points = o3d.utility.Vector3dVector(lidar_pc[:, :3])
        # with_normal = preprocess(point_cloud_original, config)
        # with_normal2.points = with_normal.points
        # with_normal2.normals = with_normal.normals
        # mesh, _ = create_mesh_from_map(
        #     None, 11, config["n_threads"], config["min_density"], with_normal2
        # )
        # lidar_pc = np.asarray(mesh.vertices, dtype=float)
        # lidar_pc = np.concatenate((lidar_pc, np.ones_like(lidar_pc[:, 0:1])), axis=1)

        lidar_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lidar_pc[:, :3]))
        # voxel downsample

        i = 0
        point_cloud_with_semantic = deepcopy(lidar_pc)

        while (
            int(i) < 10000
        ):  # Assuming the sequence does not have more than 10000 frames
            if i >= len(dict_list):
                print("finish scene!")
                break
            d = dict_list[i]

            ################## convert the static scene to the target coordinate system ##############
            # lidar_calibrated_sensor = d['lidar_calibrated_sensor']
            source_pose = d["lidar_ego_pose"]
            lidar_pcd_i = deepcopy(lidar_pcd)
            lidar_pcd_i = lidar_pcd_i.transform(pc2word2pc(target_pose, source_pose))

            point_cloud = np.asarray(lidar_pcd_i.points)

            point_cloud_with_semantic[:, :3] = point_cloud

            gt_bbox_3d = d["gt_bbox_3d"]
            locs = gt_bbox_3d[:, 0:3]
            rots = gt_bbox_3d[:, 6:7]
            # gt_bbox_3d[:, 2] += dims[:, 2] / 2.

            ################## bbox placement ##############
            object_points_list = []
            object_semantic_list = []
            for j, object_token in enumerate(d["object_tokens"]):
                for k, object_token_in_zoo in enumerate(object_token_zoo):
                    if object_token == object_token_in_zoo:
                        points = object_points_vertice[k]
                        Rot = Rotation.from_euler("z", rots[j], degrees=False)
                        rotated_object_points = Rot.apply(points)
                        points = rotated_object_points + locs[j]
                        if points.shape[0] >= 5:
                            points_in_boxes = points_in_boxes_cpu(
                                torch.from_numpy(points[:, :3][np.newaxis, :, :]),
                                torch.from_numpy(gt_bbox_3d[j : j + 1][np.newaxis, :]),
                            )
                            points = points[points_in_boxes[0, :, 0].bool()]

                        object_points_list.append(points)
                        semantics = np.ones_like(points[:, 0:1]) * object_semantic[k]
                        object_semantic_list.append(
                            np.concatenate([points[:, :3], semantics], axis=1)
                        )

            try:  # avoid concatenate an empty array
                temp = np.concatenate(object_points_list)
                scene_points = np.concatenate([point_cloud, temp])
            except:
                scene_points = point_cloud

            try:
                temp = np.concatenate(object_semantic_list)
                scene_semantic_points = np.concatenate(
                    [point_cloud_with_semantic, temp]
                )
            except:
                scene_semantic_points = point_cloud_with_semantic

            ################## remain points with a spatial range  ##############

            mask = (
                (scene_points[:, 0] > pc_range[0])
                & (scene_points[:, 0] < pc_range[3])
                & (scene_points[:, 1] > pc_range[1])
                & (scene_points[:, 1] < pc_range[4])
                & (scene_points[:, 2] > pc_range[2])
                & (scene_points[:, 2] < pc_range[5])
            )
            scene_points = scene_points[mask] - pc_range[:3]
            ################## convert points to voxels ##############
            scene_points = np.floor(scene_points / voxel_size).astype(np.int32)
            voxel = np.zeros(occ_size)
            voxel[scene_points[:, 0], scene_points[:, 1], scene_points[:, 2]] = 1

            ################## convert voxel coordinates to LiDAR system  ##############
            gt_ = voxel
            x = np.linspace(0, gt_.shape[0] - 1, gt_.shape[0])
            y = np.linspace(0, gt_.shape[1] - 1, gt_.shape[1])
            cam = np.linspace(0, gt_.shape[2] - 1, gt_.shape[2])
            X, Y, Z = np.meshgrid(x, y, cam, indexing="ij")
            vv = np.stack([X, Y, Z], axis=-1)
            fov_voxels = vv[gt_ > 0]
            fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
            fov_voxels += pc_range[:3]

            scene_semantic_points = scene_semantic_points[mask]

            ################## Nearest Neighbor to assign semantics ##############
            dense_voxels = fov_voxels
            sparse_voxels_semantic = scene_semantic_points

            x = torch.from_numpy(dense_voxels).cuda().unsqueeze(0).float()
            y = (
                torch.from_numpy(sparse_voxels_semantic[:, :3])
                .cuda()
                .unsqueeze(0)
                .float()
            )
            _, _, idx1, _ = chamfer.forward(x, y)
            indices = idx1[0].cpu().numpy()

            dense_semantic = sparse_voxels_semantic[:, 3][np.array(indices)]
            dense_voxels_with_semantic = np.concatenate(
                [fov_voxels, dense_semantic[:, np.newaxis]], axis=1
            )

            # to voxel coordinate
            dense_voxels_with_semantic[:, :3] = (
                dense_voxels_with_semantic[:, :3] - pc_range[:3]
            ) / voxel_size
            dense_voxels_with_semantic = np.floor(dense_voxels_with_semantic).astype(
                np.int32
            )
            dense_voxels_with_semantic_3dim = free_index * np.ones(occ_size)
            dense_voxels_with_semantic_3dim[
                dense_voxels_with_semantic[:, 0],
                dense_voxels_with_semantic[:, 1],
                dense_voxels_with_semantic[:, 2],
            ] = dense_voxels_with_semantic[:, 3]
            # view_3dim = np.where(view_angle_mask, dense_voxels_with_semantic_3dim, 0)
            visible_voxel = get_visible_voxels_vector(
                dense_voxels_with_semantic_3dim, voxel_center
            )
            final_point = voxel2points(
                visible_voxel, dense_voxels_with_semantic_3dim
            ).astype(np.int32)
            # mapping id
            final_point[:, 3] = class_map_array[final_point[:, 3]]
            np.save(
                os.path.join(save_path, "pts_occ_gt/" + str(start_id + i).zfill(6))
                + ".npy",
                final_point,
            )

            ##############################################################

            ##############################################################

            try:  # avoid concatenate an empty array
                temp = np.concatenate(object_points_list)
                scene_points = np.concatenate([point_cloud, temp])
            except:
                scene_points = point_cloud

            try:
                temp = np.concatenate(object_semantic_list)
                scene_semantic_points = np.concatenate(
                    [point_cloud_with_semantic, temp]
                )
            except:
                scene_semantic_points = point_cloud_with_semantic

            scene_cut_pcd = o3d.geometry.PointCloud()
            scene_cut_pcd.points = o3d.utility.Vector3dVector(scene_points)
            scene_cut_pcd.transform(os2pyl_mtx)
            scene_cut_points = np.asarray(scene_cut_pcd.points)

            cut_flag = cut_point(scene_cut_points, P_cma)
            scene_points = scene_points[cut_flag]
            scene_semantic_points = scene_semantic_points[cut_flag]
            mask = (
                (scene_points[:, 0] > pc_range[0])
                & (scene_points[:, 0] < pc_range[3])
                & (scene_points[:, 1] > pc_range[1])
                & (scene_points[:, 1] < pc_range[4])
                & (scene_points[:, 2] > pc_range[2])
                & (scene_points[:, 2] < pc_range[5])
            )
            scene_points = scene_points[mask] - pc_range[:3]
            ################## convert points to voxels ##############
            scene_points = np.floor(scene_points / voxel_size).astype(np.int32)
            voxel = np.zeros(occ_size)
            voxel[scene_points[:, 0], scene_points[:, 1], scene_points[:, 2]] = 1

            ################## convert voxel coordinates to LiDAR system  ##############
            gt_ = voxel
            x = np.linspace(0, gt_.shape[0] - 1, gt_.shape[0])
            y = np.linspace(0, gt_.shape[1] - 1, gt_.shape[1])
            cam = np.linspace(0, gt_.shape[2] - 1, gt_.shape[2])
            X, Y, Z = np.meshgrid(x, y, cam, indexing="ij")
            vv = np.stack([X, Y, Z], axis=-1)
            fov_voxels = vv[gt_ > 0]
            fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
            fov_voxels += pc_range[:3]

            scene_semantic_points = scene_semantic_points[mask]

            ################## Nearest Neighbor to assign semantics ##############
            dense_voxels = fov_voxels
            sparse_voxels_semantic = scene_semantic_points

            x = torch.from_numpy(dense_voxels).cuda().unsqueeze(0).float()
            y = (
                torch.from_numpy(sparse_voxels_semantic[:, :3])
                .cuda()
                .unsqueeze(0)
                .float()
            )
            _, _, idx1, _ = chamfer.forward(x, y)
            indices = idx1[0].cpu().numpy()

            dense_semantic = sparse_voxels_semantic[:, 3][np.array(indices)]
            dense_voxels_with_semantic = np.concatenate(
                [fov_voxels, dense_semantic[:, np.newaxis]], axis=1
            )

            # to voxel coordinate
            dense_voxels_with_semantic[:, :3] = (
                dense_voxels_with_semantic[:, :3] - pc_range[:3]
            ) / voxel_size
            dense_voxels_with_semantic = np.floor(dense_voxels_with_semantic).astype(
                np.int32
            )
            dense_voxels_with_semantic_3dim = free_index * np.ones(occ_size)
            dense_voxels_with_semantic_3dim[
                dense_voxels_with_semantic[:, 0],
                dense_voxels_with_semantic[:, 1],
                dense_voxels_with_semantic[:, 2],
            ] = dense_voxels_with_semantic[:, 3]
            # view_3dim = np.where(view_angle_mask, dense_voxels_with_semantic_3dim, 0)
            visible_voxel = get_visible_voxels_vector(
                dense_voxels_with_semantic_3dim, voxel_center_pyl
            )
            final_pyl_point = voxel2points(
                visible_voxel, dense_voxels_with_semantic_3dim
            ).astype(np.int32)
            # mapping id
            final_pyl_point[:, 3] = class_map_array[final_pyl_point[:, 3]]
            np.save(
                os.path.join(save_path, "cam_occ_gt/" + str(start_id + i).zfill(6))
                + ".npy",
                final_pyl_point,
            )
            print(
                "finish dense voxel processing %d in sequence %s"
                % (start_id + i, save_path)
            )

            i = i + 1
