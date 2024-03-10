# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import os
import sys
import time
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mmdet.visualization import DetLocalVisualizer, get_palette
from mmengine.dist import master_only
from mmengine.logging import print_log
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer as MMENGINE_Visualizer
from mmengine.visualization.utils import check_type, color_val_matplotlib, tensor2ndarray
from torch import Tensor

from mmdet3d.registry import VISUALIZERS
from mmdet3d.structures import (
    BaseInstance3DBoxes,
    Box3DMode,
    CameraInstance3DBoxes,
    Coord3DMode,
    DepthInstance3DBoxes,
    DepthPoints,
    Det3DDataSample,
    LiDARInstance3DBoxes,
    PointData,
    points_cam2img,
)
from mmdet3d.visualization.vis_utils import (
    proj_camera_bbox3d_to_img,
    proj_depth_bbox3d_to_img,
    proj_lidar_bbox3d_to_img,
    to_depth_mode,
)
from mmdet3d.visualization.local_visualizer import Det3DLocalVisualizer

try:
    import open3d as o3d
    from open3d import geometry
    from open3d.visualization import Visualizer
except ImportError:
    o3d = geometry = Visualizer = None


@VISUALIZERS.register_module()
class OccLocalVisualizer(Det3DLocalVisualizer):
    """MMDetection3D Local Visualizer.

    - 3D detection and segmentation drawing methods

      - draw_bboxes_3d: draw 3D bounding boxes on point clouds
      - draw_proj_bboxes_3d: draw projected 3D bounding boxes on image
      - draw_seg_mask: draw segmentation mask via per-point colorization

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        points (np.ndarray, optional): Points to visualize with shape (N, 3+C).
            Defaults to None.
        image (np.ndarray, optional): The origin image to draw. The format
            should be RGB. Defaults to None.
        pcd_mode (int): The point cloud mode (coordinates): 0 represents LiDAR,
            1 represents CAMERA, 2 represents Depth. Defaults to 0.
        vis_backends (List[dict], optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
            Defaults to None.
        bbox_color (str or Tuple[int], optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to None.
        text_color (str or Tuple[int]): Color of texts. The tuple of color
            should be in BGR order. Defaults to (200, 200, 200).
        mask_color (str or Tuple[int], optional): Color of masks. The tuple of
            color should be in BGR order. Defaults to None.
        line_width (int or float): The linewidth of lines. Defaults to 3.
        frame_cfg (dict): The coordinate frame config while Open3D
            visualization initialization.
            Defaults to dict(size=1, origin=[0, 0, 0]).
        alpha (int or float): The transparency of bboxes or mask.
            Defaults to 0.8.
        multi_imgs_col (int): The number of columns in arrangement when showing
            multi-view images.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from mmengine.structures import InstanceData
        >>> from mmdet3d.structures import (DepthInstance3DBoxes
        ...                                 Det3DDataSample)
        >>> from mmdet3d.visualization import Det3DLocalVisualizer

        >>> det3d_local_visualizer = Det3DLocalVisualizer()
        >>> image = np.random.randint(0, 256, size=(10, 12, 3)).astype('uint8')
        >>> points = np.random.rand(1000, 3)
        >>> gt_instances_3d = InstanceData()
        >>> gt_instances_3d.bboxes_3d = DepthInstance3DBoxes(
        ...     torch.rand((5, 7)))
        >>> gt_instances_3d.labels_3d = torch.randint(0, 2, (5,))
        >>> gt_det3d_data_sample = Det3DDataSample()
        >>> gt_det3d_data_sample.gt_instances_3d = gt_instances_3d
        >>> data_input = dict(img=image, points=points)
        >>> det3d_local_visualizer.add_datasample('3D Scene', data_input,
        ...                                       gt_det3d_data_sample)

        >>> from mmdet3d.structures import PointData
        >>> det3d_local_visualizer = Det3DLocalVisualizer()
        >>> points = np.random.rand(1000, 3)
        >>> gt_pts_seg = PointData()
        >>> gt_pts_seg.pts_semantic_mask = torch.randint(0, 10, (1000, ))
        >>> gt_det3d_data_sample = Det3DDataSample()
        >>> gt_det3d_data_sample.gt_pts_seg = gt_pts_seg
        >>> data_input = dict(points=points)
        >>> det3d_local_visualizer.add_datasample('3D Scene', data_input,
        ...                                       gt_det3d_data_sample,
        ...                                       vis_task='lidar_seg')
    """

    def __init__(
        self,
        name: str = "visualizer",
        points: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        pcd_mode: int = 0,
        vis_backends: Optional[List[dict]] = None,
        save_dir: Optional[str] = None,
        ssc_show_dir: Optional[str] = None,
        bbox_color: Optional[Union[str, Tuple[int]]] = None,
        text_color: Union[str, Tuple[int]] = (200, 200, 200),
        mask_color: Optional[Union[str, Tuple[int]]] = None,
        line_width: Union[int, float] = 3,
        frame_cfg: dict = dict(size=1, origin=[0, 0, 0]),
        alpha: Union[int, float] = 0.8,
        multi_imgs_col: int = 3,
        fig_show_cfg: dict = dict(figsize=(18, 12)),
    ) -> None:
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            line_width=line_width,
            alpha=alpha,
        )
        if points is not None:
            self.set_points(points, pcd_mode=pcd_mode, frame_cfg=frame_cfg)
        self.multi_imgs_col = multi_imgs_col
        self.fig_show_cfg.update(fig_show_cfg)

        self.flag_pause = False
        self.flag_next = False
        self.flag_exit = False
        self.ssc_show_dir = ssc_show_dir

    @master_only
    def add_datasample(
        self,
        name: str,
        data_input: dict,
        data_sample: Optional[Det3DDataSample] = None,
        draw_gt: bool = True,
        draw_pred: bool = True,
        show: bool = False,
        wait_time: float = 0,
        out_file: Optional[str] = None,
        o3d_save_path: Optional[str] = None,
        vis_task: str = "mono_det",
        pred_score_thr: float = 0.3,
        step: int = 0,
        show_pcd_rgb: bool = False,
    ) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are displayed
          in a stitched image where the left image is the ground truth and the
          right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and the images
          will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be saved to
          ``out_file``. It is usually used when the display is not available.

        Args:
            name (str): The image identifier.
            data_input (dict): It should include the point clouds or image
                to draw.
            data_sample (:obj:`Det3DDataSample`, optional): Prediction
                Det3DDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT Det3DDataSample.
                Defaults to True.
            draw_pred (bool): Whether to draw Prediction Det3DDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn point clouds and image.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str, optional): Path to output file. Defaults to None.
            o3d_save_path (str, optional): Path to save open3d visualized
                results. Defaults to None.
            vis_task (str): Visualization task. Defaults to 'mono_det'.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
            show_pcd_rgb (bool): Whether to show RGB point cloud. Defaults to
                False.
        """
        assert vis_task in ("mono_det", "multi-view_det", "lidar_det", "lidar_seg", "multi-modality_det"), f"got unexpected vis_task {vis_task}."
        classes = self.dataset_meta.get("classes", None)
        # For object detection datasets, no palette is saved
        palette = self.dataset_meta.get("palette", None)
        ignore_index = self.dataset_meta.get("ignore_index", None)
        if vis_task == "lidar_seg" and ignore_index is not None and "pts_semantic_mask" in data_sample.gt_pts_seg:  # noqa: E501
            keep_index = data_sample.gt_pts_seg.pts_semantic_mask != ignore_index  # noqa: E501
        else:
            keep_index = None

        gt_data_3d = None
        pred_data_3d = None
        gt_img_data = None
        pred_img_data = None

        if not hasattr(self, "o3d_vis") and vis_task in ["multi-view_det", "lidar_det", "lidar_seg", "multi-modality_det"]:
            self.o3d_vis = self._initialize_o3d_vis(show=show)

        if draw_pred and data_sample is not None:
            if "pred_pts_seg" in data_sample and vis_task == "lidar_seg":
                assert classes is not None, "class information is " "not provided when " "visualizing semantic " "segmentation results."
                assert "points" in data_input
                self._draw_pts_sem_seg(data_sample.pred_pts_geo, data_sample.pred_pts_seg, palette, keep_index)

                if self.ssc_show_dir is not None:
                    pc = np.concatenate(
                        [data_sample.pred_pts_geo, data_sample.pred_pts_seg.pts_semantic_mask.reshape(-1, 1)],
                        axis=1,
                    )
                    np.savetxt(os.path.join(self.ssc_show_dir, name[:-4] + ".txt"), pc, fmt="%.6f")

        # monocular 3d object detection image
        if vis_task in ["mono_det", "multi-modality_det"]:
            if gt_data_3d is not None and pred_data_3d is not None:
                drawn_img_3d = np.concatenate((gt_data_3d["img"], pred_data_3d["img"]), axis=1)
            elif gt_data_3d is not None:
                drawn_img_3d = gt_data_3d["img"]
            elif pred_data_3d is not None:
                drawn_img_3d = pred_data_3d["img"]
            else:  # both instances of gt and pred are empty
                drawn_img_3d = None
        else:
            drawn_img_3d = None

        # 2d object detection image
        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            drawn_img = None

        if show:
            self.show(o3d_save_path, drawn_img_3d, drawn_img, win_name=name, wait_time=wait_time, vis_task=vis_task)

        if out_file is not None:
            # check the suffix of the name of image file
            if not (out_file.endswith(".png") or out_file.endswith(".jpg")):
                out_file = f"{out_file}.png"
            if drawn_img_3d is not None:
                mmcv.imwrite(drawn_img_3d[..., ::-1], out_file)
            if drawn_img is not None:
                mmcv.imwrite(drawn_img[..., ::-1], out_file[:-4] + "_2d" + out_file[-4:])
        else:
            self.add_image(name, drawn_img_3d, step)
