from mmengine.config import read_base
from mmengine.visualization.vis_backend import WandbVisBackend, LocalVisBackend
from mmdet3d.visualization.local_visualizer import Det3DLocalVisualizer


with read_base():
    from mmdet3d.configs._base_.default_runtime import *
    from .base.cosine import *
    from .base.net import *
    from .base.semankitti import *

# train_dataloader.update(dataset=dict(indices=50))
# val_dataloader.update(dataset=dict(indices=50))

# dataset settings
class_names = (
    "free",
    "occupied",
)
palette = list(
    [
        [0, 0, 0],
        [255, 255, 255],
    ]
)

labels_map = {
    0: 0,
    1: 1,
}
metainfo = dict(classes=class_names, seg_label_mapping=labels_map, max_label=259)
train_pipeline = [
    dict(type=LoadPointsFromFile, coord_type="LIDAR", load_dim=4, use_dim=4, backend_args=backend_args),
    dict(
        type=LoadVoxelLabelFromFile,
        task="sc",
        scale=scale,
        ignore_index=ignore_index,
        grid_size=grid_size,
    ),
    dict(
        type=ApplayVisMask,
        center=[0, 0, 0],
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        fov=fov_vertical,
    ),
    dict(
        type=PackSscInputs,
        keys=["points", "voxel_label"],
    ),
]

test_pipeline = [
    dict(type=LoadPointsFromFile, coord_type="LIDAR", load_dim=4, use_dim=4, backend_args=backend_args),
    dict(
        type=LoadVoxelLabelFromFile,
        task="sc",
        scale=scale,
        ignore_index=ignore_index,
        grid_size=grid_size,
    ),
    dict(
        type=PackSscInputs,
        keys=["points", "voxel_label"],
    ),
]

val_pipeline = train_pipeline

train_split = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file="semantickittiDataset_infos_train.pkl",
    pipeline=train_pipeline,
    metainfo=metainfo,
    modality=input_modality,
    backend_args=backend_args,
    ignore_index=ignore_index,
)

val_split = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file="semantickittiDataset_infos_val.pkl",
    pipeline=val_pipeline,
    metainfo=metainfo,
    modality=input_modality,
    test_mode=True,
    backend_args=backend_args,
    ignore_index=ignore_index,
)

test_split = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file="semantickittiDataset_infos_test.pkl.pkl",
    pipeline=test_pipeline,
    metainfo=metainfo,
    modality=input_modality,
    test_mode=True,
    backend_args=backend_args,
    ignore_index=ignore_index,
)

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=train_split,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=val_split,
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=test_split,
)
# visualization settings
vis_backends = [
    dict(type=LocalVisBackend),
    dict(type=WandbVisBackend, init_kwargs=dict(project="sc", name="32b8convfuse-refGT")),
]
visualizer = dict(type=Det3DLocalVisualizer, vis_backends=vis_backends, name="visualizer")

# checkpoint_config = None
checkpoint_config = dict(interval=1)
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=24, val_interval=1)

# model settings
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
        range_img=True,
        range_layer=dict(
            H=64,
            W=1024,
            fov_up=3.0,
            fov_down=-25.0,
            fov_left=-90.0,
            fov_right=90.0,
            means=(11.71279, -0.1023471, 0.4952, -1.0545, 0.2877),
            stds=(10.24, 12.295865, 9.4287, 0.8643, 0.1450),
        ),
    ),
    pts_bev_backbone=dict(
        type=BevNet,
        act_cfg=HSwin,
    ),
    pts_range_backbone=dict(
        type=RangeNet,
        in_channels=5,
        stem_channels=range_encoder_channel,
        num_stages=3,
        stage_blocks=(2, 2, 2),
        out_channels=(
            range_encoder_channel,
            range_encoder_channel,
            range_encoder_channel,
        ),
        strides=(2, 2, 2),
        dilations=(1, 1, 1),
        fuse_channels=(
            range_encoder_channel,
            fuse_channel,
        ),
        act_cfg=HSwin,
    ),
    pts_fusion_neck=dict(
        type=FusionNet,
        conv_cfg=dict(type=nn.Conv2d),
        norm_cfg=dict(type=nn.BatchNorm2d),
    ),
    pts_ssc_head=dict(
        type=DenseSscHead,
        inplanes=fuse_channel,
        planes=fuse_channel,
        nbr_classes=number_classes,
        dilations_conv_list=[1, 2, 3],
        loss_ce=dict(
            type=CrossEntropyLoss,
            class_weight=[0.45, 0.55],
        ),
        ignore_index=255,
        conv_cfg=dict(type=nn.Conv3d),
        norm_cfg=dict(type=nn.BatchNorm3d),
        act_cfg=HSwin,
    ),
)
randomness = dict(seed=3407)
# auto_scale_lr = dict(enable=False, base_batch_size=2)
