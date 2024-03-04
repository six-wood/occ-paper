import torch.nn as nn
from mmengine.config import read_base
from mmengine.visualization.vis_backend import WandbVisBackend, LocalVisBackend
from mmdet.models.losses import CrossEntropyLoss
from mmdet3d.visualization.local_visualizer import Det3DLocalVisualizer
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from projects.occ_paper.mmdet3d_plugin.models.sc_net import ScNet

with read_base():
    from mmdet3d.configs._base_.default_runtime import *
    from .base.lr import *
    from .base.semankitti import *

model = dict(
    type=ScNet,
    use_pred_mask=True,
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
    free_index=0,
    ignore_index=255,
    loss_sc=dict(
        type=CrossEntropyLoss,
        class_weight=[0.446, 0.505],
        loss_weight=1.0,
    ),
    norm_cfg=dict(type=nn.SyncBatchNorm),
    act_cfg=dict(type=nn.Hardswish, inplace=True),
)

custom_imports = dict(imports=["projects.occ_paper.mmdet3d_plugin"], allow_failed_imports=False)
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

val_pipeline = train_pipeline

train_split.update(dict(metainfo=metainfo, pipeline=train_pipeline))
val_split.update(dict(metainfo=metainfo, pipeline=val_pipeline))
test_split.update(dict(metainfo=metainfo, pipeline=test_pipeline))

train_dataloader.update(dict(dataset=train_split, batch_size=4))
val_dataloader.update(dict(dataset=val_split, batch_size=4))
test_dataloader.update(dict(dataset=test_split, batch_size=4))

# visualization settings
vis_backends = [
    dict(type=LocalVisBackend),
    dict(type=WandbVisBackend, init_kwargs=dict(project="sc+seg", name="baseline")),
]
visualizer = dict(type=Det3DLocalVisualizer, vis_backends=vis_backends, name="visualizer")

val_evaluator = dict(type=SscMetric)
test_evaluator = val_evaluator
# debug settings
# train_dataloader.update(dataset=dict(indices=100))
# val_dataloader.update(dataset=dict(indices=50))
# train_dataloader.update(batch_size=1)
# val_dataloader.update(batch_size=1)
# test_dataloader.update(batch_size=1)
