from dense_voxel import dense_voxel_process
from pathlib import Path
import open3d as o3d
import numpy as np
import yaml
import copy


class trans_merge(object):
    def __init__(self, home_path: str, save_path: str):
        self.home_path = home_path
        self.save_path = save_path
        seq = ["00000", "00001", "00002", "00003", "00004"]
        self.seqs = [self.home_path + "/" + s for s in seq]
        self.save_seqs = [self.save_path + "/" + s for s in seq]

        self.mot = [self.save_path + "/" + s + "/" + s + ".txt" for s in seq]

        vel = "vel_cloud_node_kitti_bin"
        self.vels = [self.home_path + "/" + s + "/" + vel for s in seq]

        vel_label = "vel_cloud_node_semantickitti_label_id"
        self.vels_label = [self.home_path + "/" + s + "/" + vel_label for s in seq]

        os = "os1_cloud_node_kitti_bin"
        self.os = [self.home_path + "/" + s + "/" + os for s in seq]

        os_label = "os1_cloud_node_semantickitti_label_id"
        self.os_label = [self.home_path + "/" + s + "/" + os_label for s in seq]

        bbox = "bbox"
        self.bboxs = [self.save_path + "/" + s + "/" + bbox + "/" for s in seq]

        pc = "pc"
        self.pcs = [self.save_path + "/" + s + "/" + pc + "/" for s in seq]

        pc_label = "pc_label"
        self.pc_labels = [self.save_path + "/" + s + "/" + pc_label + "/" for s in seq]

        cut = "cut"
        self.cuts = [self.save_path + "/" + s + "/" + cut + "/" for s in seq]

        cut_label = "cut_label"
        self.cut_labels = [
            self.save_path + "/" + s + "/" + cut_label + "/" for s in seq
        ]

        pts_occ_gt = "pts_occ_gt"
        self.sence_gts = [
            self.save_path + "/" + s + "/" + pts_occ_gt + "/" for s in seq
        ]

        cams_occ_gt = "cam_occ_gt"
        self.cams_gts = [
            self.save_path + "/" + s + "/" + cams_occ_gt + "/" for s in seq
        ]

        self.pox_file = [self.home_path + "/" + s + "/poses.txt" for s in seq]
        self.pos_info = {
            seq[0]: np.loadtxt(self.pox_file[0]),
            seq[1]: np.loadtxt(self.pox_file[1]),
            seq[2]: np.loadtxt(self.pox_file[2]),
            seq[3]: np.loadtxt(self.pox_file[3]),
            seq[4]: np.loadtxt(self.pox_file[4]),
        }

        for m in zip(
            self.bboxs,
            self.sence_gts,
            self.cams_gts,
            self.pcs,
            self.pc_labels,
            self.cuts,
            self.cut_labels,
        ):
            for i in m:
                p = Path(i)
                if not p.is_dir():
                    p.mkdir(parents=True, exist_ok=True)

        sample_num = {
            seq[0]: 2847,
            seq[1]: 2319,
            seq[2]: 4147,
            seq[3]: 2184,
            seq[4]: 2059,
        }

        self.sample_num = [sample_num[s] for s in seq]

        camera_info = [2813.643275, 2808.326079, 969.285772, 624.049972]
        P = np.ones((3, 3))
        P[0, 0] = camera_info[0]
        P[1, 1] = camera_info[1]
        P[2, 2] = 1
        P[0, 2] = camera_info[2]
        P[1, 2] = camera_info[3]
        self.P = P

        self.person_label = 17
        self.void_flag = 0

        move_range = {
            seq[0]: [120, 2600],
            seq[1]: [60, 2180],
            seq[2]: [20, 4120],
            seq[3]: [20, 1980],
            seq[4]: [120, 1980],
        }
        self.move_range = [move_range[s] for s in seq]

        cut_range = {
            seq[0]: [[-5, -5, -100], [2, 2, 1.5]],
            seq[1]: [[-3, -3, -100], [3, 3, 1.5]],
            seq[2]: [[-5, -5, -100], [2, 2, 1.5]],
            seq[3]: [[-5, -5, -100], [2, 2, 1.5]],
            seq[4]: [[-5, -5, -100], [2, 2, 1.5]],
        }

        self.self_range = [cut_range[s] for s in seq]

        self.config_path = "/home/lms/code/ContrasOcc/tools/config.yaml"
        with open(self.config_path, "r") as stream:
            config = yaml.safe_load(stream)
        pc_range = config["pc_range"]
        self.pc_range = np.array(pc_range)

        self.vis = o3d.visualization.Visualizer()

    def os2cam(self, filepath, key):
        with open(filepath, "r") as f:
            data = yaml.load(f, Loader=yaml.Loader)
        q = data[key]["q"]
        q_wxyz = np.array([q["w"], q["x"], q["y"], q["z"]])
        t = data[key]["t"]
        t = np.array([t["x"], t["y"], t["z"]])
        R = o3d.geometry.get_rotation_matrix_from_quaternion(q_wxyz)

        R_ = R.T
        t_ = -R_ @ t

        RT = np.eye(4, 4)
        RT[:3, :3] = R_
        RT[:3, -1] = t_
        return RT

    def vel2os(self, filepath, key):
        with open(filepath, "r") as f:
            data = yaml.load(f, Loader=yaml.Loader)
        q = data[key]["q"]
        q_wxyz = np.array([q["w"], q["x"], q["y"], q["z"]])
        t = data[key]["t"]
        t = np.array([t["x"], t["y"], t["z"]])
        R = o3d.geometry.get_rotation_matrix_from_quaternion(q_wxyz)

        R_ = R.T
        t_ = -R_ @ t

        RT = np.eye(4, 4)
        RT[:3, :3] = R_
        RT[:3, -1] = t_
        return RT

    # cut the point cloud by P

    def cut_point(
        self, points, label, pad_x=0, pad_y=0, img_width=1920, img_height=1200
    ):
        P = self.P
        fov_x = 2 * np.arctan2(img_width, 2 * P[0, 0]) * 180 / np.pi + pad_x
        fov_y = 2 * np.arctan2(img_height, 2 * P[1, 1]) * 180 / np.pi + pad_y
        R = np.eye(4)
        p_l = np.ones((points.shape[0], points.shape[1] + 1))
        p_l[:, :3] = points[:, :3].copy()
        x = p_l[:, 0]
        y = p_l[:, 1]
        z = p_l[:, 2]
        xangle = np.arctan2(x, z) * 180 / np.pi
        yangle = np.arctan2(y, z) * 180 / np.pi
        flag2 = (xangle > -fov_x / 2) & (xangle < fov_x / 2)
        flag3 = (yangle > -fov_y / 2) & (yangle < fov_y / 2)
        res = p_l[flag2 & flag3, :3]
        return points[flag2 & flag3, :], label[flag2 & flag3], flag2 & flag3

    def load_pc(self, filepath):
        point = np.fromfile(filepath, dtype=np.float32)
        point = point.reshape(-1, 4)
        return point

    def load_label(self, filepath):
        label = np.fromfile(filepath, dtype=np.uint32)
        label = label.reshape(-1)
        return label

    def save_bin(self, filepath, point):
        point = point.reshape(-1)
        point.astype(np.float32).tofile(filepath)

    def save_label(self, filepath, label):
        label = label.reshape(-1)
        label.astype(np.uint32).tofile(filepath)

    def save_txt(self, filepath, point):
        np.savetxt(filepath, point, fmt="%f")

    def transform_point(self, point, mtx):
        point_ = np.ones((point.shape[0], 4))
        point_[:, :3] = point[:, :3]
        point_ = mtx @ point_.T
        point_ = point_.T[:, :3]
        point_copy = point.copy()
        point_copy[:, :3] = point_
        return point_copy

    def generate_3dbox(self, box_2d, points_pyl, points, label):
        x_min, y_min, width, height = box_2d
        x = points_pyl[:, 0]
        y = points_pyl[:, 1]
        z = points_pyl[:, 2]
        x_2d = self.P[0, 0] * (x / z) + self.P[0, 2]
        y_2d = self.P[1, 1] * (y / z) + self.P[1, 2]
        in_box = (
            (x_2d > x_min)
            & (x_2d < x_min + width)
            & (y_2d > y_min)
            & (y_2d < y_min + height)
        )
        points_in_box = points[in_box]
        label = label[in_box]
        in_box = label == self.person_label
        points_in_box = points_in_box[in_box]
        box_3d = np.zeros((1, 7))
        if points_in_box.shape[0] > 0:
            box_3d = np.zeros((1, 7))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_in_box[:, :3])
            min_bound = pcd.get_min_bound()
            max_bound = pcd.get_max_bound()
            bottom_center = (min_bound + max_bound) / 2
            bottom_center[2] = min_bound[2]
            box_size = max_bound - min_bound
            box_3d[0, :3] = bottom_center
            box_3d[0, 3:6] = box_size

        return box_3d, points_pyl

    def process_bin_points(self, seq_id: int):
        pc_min = self.pc_range[:3]
        pc_max = self.pc_range[3:]
        for i in range(0, self.sample_num[seq_id]):
            os = self.load_pc(self.os[seq_id] + "/" + str(i).zfill(6) + ".bin")
            os_label = self.load_label(
                self.os_label[seq_id] + "/" + str(i).zfill(6) + ".label"
            )

            # filter x y z = [0,0,0]
            flag = ~np.all(os[:, :3] == 0, axis=1)
            os = os[flag]
            os_label = os_label[flag]

            # save pc
            self.save_bin(
                self.pcs[seq_id] + "/" + str(i).zfill(6) + ".bin",
                os,
            )

            self.save_label(
                self.pc_labels[seq_id] + "/" + str(i).zfill(6) + ".label",
                os_label,
            )

            mask = (
                (os[:, 0] > pc_min[0])
                & (os[:, 0] < pc_max[0])
                & (os[:, 1] > pc_min[1])
                & (os[:, 1] < pc_max[1])
                & (os[:, 2] > pc_min[2])
                & (os[:, 2] < pc_max[2])
            )

            cut = os[mask]
            cut_label = os_label[mask]

            # pcd.points = o3d.utility.Vector3dVector(os[:, :3])
            # pcd = pcd.transform(os2pyl_mtx)
            # os2pyl = np.concatenate((np.asarray(pcd.points), os[:, 3:]), axis=1)

            # cut, cut_label, _ = self.cut_point(os2pyl, os_label)

            self.save_bin(
                self.cuts[seq_id] + "/" + str(i).zfill(6) + ".bin",
                cut,
            )

            self.save_label(
                self.cut_labels[seq_id] + "/" + str(i).zfill(6) + ".label",
                cut_label,
            )
            print("process seq: %s, sample: %d" % (self.seqs[seq_id], i))

    # def bin2txt(self, id: int):
    #     for i in range(0, self.sample_num[id]):
    #         cut = self.load_pc(self.cuts[id] + "/" + str(i).zfill(6) + ".bin")
    #         label = self.load_label(
    #             self.cut_labels[id] + "/" + str(i).zfill(6) + ".label"
    #         )

    #         label = np.float32(label)

    #         merge_cut = np.concatenate((cut, label.reshape(-1, 1)), axis=1)

    #         self.save_txt(
    #             self.cut_show[id] + "/" + str(i).zfill(6) + ".txt",
    #             merge_cut,
    #         )

    #         print("process seq: %s, sample: %d" % (self.cut_show[id], i))

    # filter person ,generate person bbox ,id
    def generate_bbox(self, seq_id: int):
        os2pyl_mtx = self.os2cam(
            self.seqs[seq_id] + "/transforms.yaml",
            "os1_cloud_node-pylon_camera_node",
        )
        pcd = o3d.geometry.PointCloud()

        with open(self.mot[seq_id], "r") as f:
            boxs_info = f.readlines()
            boxs_info = [x.strip().split(" ") for x in boxs_info]
        frame_to_box_map = {}
        for data in boxs_info:
            frame = int(data[0])
            box_id = int(data[1])
            box_coordinates = list(map(float, data[2:6]))  # 将坐标转换为浮点数

            # 组合box_id和坐标
            box_info = {"id": box_id, "coordinates": box_coordinates}

            if frame not in frame_to_box_map:
                frame_to_box_map[frame] = []
            frame_to_box_map[frame].append(box_info)

        self.vis.create_window(window_name="Open3D", width=800, height=600)
        for i in range(0, self.sample_num[seq_id]):
            pc = self.load_pc(self.pcs[seq_id] + "/" + str(i).zfill(6) + ".bin")
            pc_lable = self.load_label(
                self.pc_labels[seq_id] + "/" + str(i).zfill(6) + ".label"
            )
            pc_copy = copy.deepcopy(pc)
            pcd.points = o3d.utility.Vector3dVector(pc_copy[:, :3])
            pcd = pcd.transform(os2pyl_mtx)
            pc_pyl = np.concatenate((np.asarray(pcd.points), pc_copy[:, 3:]), axis=1)
            person_flag = pc_lable == self.person_label

            pcd = o3d.geometry.PointCloud()
            bbox = np.zeros((0, 8))
            person_frame = np.zeros((0, 4))
            if i in frame_to_box_map:
                for box_info in frame_to_box_map[i]:
                    box_id = box_info["id"]
                    box_coordinates = box_info["coordinates"]
                    box_2d = box_coordinates
                    box_3d, person_pc = self.generate_3dbox(
                        box_2d, pc_pyl, pc, pc_lable
                    )
                    person_frame = np.concatenate((person_frame, person_pc), axis=0)
                    box_3d = np.concatenate((box_3d, np.array([[box_id]])), axis=1)
                    bbox = np.concatenate((bbox, box_3d), axis=0)

                # 运行可视化窗口
                person_frame[:, 2], person_frame[:, 3] = (
                    person_frame[:, 3],
                    person_frame[:, 2].copy(),
                )
                person_frame[:, 1] = -person_frame[:, 1]
                pcd.points = o3d.utility.Vector3dVector(person_frame[:, :3])
                self.vis.clear_geometries()
                self.vis.add_geometry(pcd)
                self.vis.poll_events()
                self.vis.update_renderer()

            # pc = pc[~person_flag]
            # pc_lable = pc_lable[~person_flag]
            # pc = np.concatenate((pc, person_frame), axis=0)
            # append_label = np.ones((person_frame.shape[0], 1)) * self.person_label
            # append_label = append_label.reshape(-1).astype(np.uint32)
            # pc_lable = np.concatenate((pc_lable, append_label), axis=0)

            # save cut
            # self.save_bin(
            #     self.cuts[seq_id] + "/" + str(i).zfill(6) + ".bin",
            #     cut,
            # )
            # self.save_label(
            #     self.cut_labels[seq_id] + "/" + str(i).zfill(6) + ".label",
            #     cut_lable,
            # )

            category = np.ones((bbox.shape[0], 1)) * self.person_label
            box_size = np.zeros((bbox.shape[0], 3))
            box_size[:, 0] = bbox[:, 4]
            box_size[:, 1] = bbox[:, 5]
            box_size[:, 2] = bbox[:, 3]

            # save bbox

            np.save(
                self.bboxs[seq_id] + "bbox" + str(i) + ".npy",
                bbox[:, :7].astype(np.float32),
            )

            np.save(
                self.bboxs[seq_id] + "object_category" + str(i) + ".npy",
                category.astype(np.int32),
            )

            np.save(
                self.bboxs[seq_id] + "boxes_token" + str(i) + ".npy",
                np.char.array(bbox[:, 7]).astype("<U32"),
            )

            print("process seq bbox: %s, sample: %d" % (self.seqs[seq_id], i))


t = trans_merge("/data/Rellis-3D", "/home/lms/code/dataBuffer/Rellis3D")
for i in [0, 1, 2, 3, 4]:
    t.process_bin_points(i)

for i in [0, 1, 2, 3, 4]:
    t.generate_bbox(i)
t.vis.destroy_window()

for i in [0, 1, 2, 3, 4]:
    dense_voxel_process(t.save_seqs[i], t.move_range[i], t.config_path)
