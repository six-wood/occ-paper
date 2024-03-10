from mmengine.config import read_base
from mmengine.visualization.vis_backend import WandbVisBackend, LocalVisBackend
from mmdet3d.visualization.local_visualizer import Det3DLocalVisualizer


with read_base():
    from mmdet3d.configs._base_.default_runtime import *
    from .base.lr import *
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
    dict(
        type=LoadPointsFromFile,
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(
        type=LoadImageFromFile,
        backend_args=backend_args,
    ),
    dict(
        type=LoadVoxelLabelFromFile,
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

val_pipeline = [
    dict(
        type=LoadPointsFromFile,
        coord_type="LIDAR",
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(
        type=LoadImageFromFile,
        backend_args=backend_args,
    ),
    dict(
        type=LoadVoxelLabelFromFile,
        scale=scale,
        ignore_index=ignore_index,
        grid_size=grid_size,
    ),
    dict(
        type=PackSscInputs,
        keys=["points", "voxel_label"],
    ),
]

train_split.update(dict(metainfo=metainfo, pipeline=train_pipeline))
val_split.update(dict(metainfo=metainfo, pipeline=val_pipeline))
test_split.update(dict(metainfo=metainfo, pipeline=test_pipeline))

train_dataloader.update(dict(dataset=train_split, batch_size=4))
val_dataloader.update(dict(dataset=val_split, batch_size=4))
test_dataloader.update(dict(dataset=test_split, batch_size=4))

# visualization settings
vis_backends = [
    dict(type=LocalVisBackend),
    # dict(type=WandbVisBackend, init_kwargs=dict(project="sc", name="baseline")),
]
visualizer = dict(type=Det3DLocalVisualizer, vis_backends=vis_backends, name="visualizer")

# debug settings
# train_dataloader.update(dataset=dict(indices=100))
# val_dataloader.update(dataset=dict(indices=50))
# train_dataloader.update(batch_size=1)
# val_dataloader.update(batch_size=1)
# test_dataloader.update(batch_size=1)
