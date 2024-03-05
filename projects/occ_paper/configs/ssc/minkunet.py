from mmengine.config import read_base
from mmengine.visualization.vis_backend import WandbVisBackend, LocalVisBackend
from projects.occ_paper.mmdet3d_plugin.visualization.local_visualizer import OccLocalVisualizer
from mmdet3d.datasets.transforms.transforms_3d import RandomFlip3D, GlobalRotScaleTrans
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from projects.occ_paper.mmdet3d_plugin.datasets.transforms.loading import LoadScAnnotations3D, LoadScPointsFromFile
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss

with read_base():
    from mmdet3d.configs._base_.default_runtime import *
    from mmdet3d.configs._base_.models.minkunet import *
    from ..base.lr import *
    from ..base.semankitti import *
    from ..base.share_paramenter import *

# TODO Add the original point cloud voxel to sc (for samll target: person and so on)
# TODO implement(copy) the focal loss in head
# TODO Add image RandomFlip3D
# TODO Use the cross attention fuse image and range feature

model = dict(
    type=MinkUNet,
    data_preprocessor=dict(
        type=Det3DDataPreprocessor,
        voxel=True,
        voxel_type="minkunet",
        batch_first=False,
        max_voxels=None,
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
            voxel_size=[0.2, 0.2, 0.2],
            max_voxels=(-1, -1),
        ),
    ),
    backbone=dict(
        type=MinkUNetBackbone,
        in_channels=3,
        num_stages=4,
        base_channels=32,
        encoder_channels=[32, 64, 128, 256],
        encoder_blocks=[2, 3, 4, 6],
        decoder_channels=[256, 128, 96, 96],
        decoder_blocks=[2, 2, 2, 2],
        block_type="basic",
        sparseconv_backend="torchsparse",
    ),
    decode_head=dict(
        type=MinkUNetHead,
        channels=96,
        num_classes=20,
        dropout_ratio=0,
        loss_decode=dict(
            type=CrossEntropyLoss,
            class_weight=semantickitti_class_weight,
            avg_non_ignore=True,
        ),
        ignore_index=255,
    ),
    train_cfg=dict(),
    test_cfg=dict(),
)

train_pipeline = [
    dict(type=LoadScPointsFromFile, coord_type="LIDAR", load_dim=3, use_dim=3, backend_args=backend_args),
    dict(
        type=LoadScAnnotations3D,
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype="np.int32",
        seg_offset=2**16,
        dataset_type="semantickitti",
        backend_args=backend_args,
    ),
    dict(type=RandomFlip3D, sync_2d=False, flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    dict(
        type=GlobalRotScaleTrans,
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(type=Pack3DDetInputs, keys=["points", "pts_semantic_mask"]),
]

val_pipeline = [
    dict(type=LoadScPointsFromFile, coord_type="LIDAR", load_dim=3, use_dim=3, backend_args=backend_args),
    dict(
        type=LoadScAnnotations3D,
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype="np.int32",
        seg_offset=2**16,
        dataset_type="semantickitti",
        backend_args=backend_args,
    ),
    dict(type=Pack3DDetInputs, keys=["points"]),
]

test_pipeline = [
    dict(type=LoadScPointsFromFile, coord_type="LIDAR", load_dim=3, use_dim=3, backend_args=backend_args),
    dict(type=Pack3DDetInputs, keys=["points"]),
]

train_split.update(dict(metainfo=metainfo, pipeline=train_pipeline))
val_split.update(dict(metainfo=metainfo, pipeline=val_pipeline))

train_dataloader.update(dict(dataset=train_split, batch_size=4))
val_dataloader.update(dict(dataset=val_split, batch_size=4))

# # debug settings
# train_dataloader.update(dataset=dict(indices=50))
# val_dataloader.update(dataset=dict(indices=50))

# visualization settings
vis_backends = [
    dict(type=LocalVisBackend),
    # dict(type=WandbVisBackend, init_kwargs=dict(project="ssc-topk-fuse", name="minkunet-occ")),
]
visualizer = dict(type=OccLocalVisualizer, vis_backends=vis_backends, name="visualizer", ssc_show_dir="outputs/visualizer")

val_evaluator = dict(type=SegMetric)
test_evaluator = val_evaluator
custom_imports = dict(imports=["projects.occ_paper.mmdet3d_plugin"], allow_failed_imports=False)
