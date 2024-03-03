from mmengine.config import read_base
from mmengine.visualization.vis_backend import WandbVisBackend, LocalVisBackend
from projects.occ_paper.mmdet3d_plugin.visualization.local_visualizer import OccLocalVisualizer


with read_base():
    from mmdet3d.configs._base_.default_runtime import *
    from .base.lr import *
    from .base.net import *
    from .base.semankitti import *

# TODO Add the original point cloud voxel to sc (for samll target: person and so on)
# TODO implement(copy) the focal loss in head
# TODO Add image RandomFlip3D
# TODO Use the cross attention fuse image and range feature

# train settings
train_dataloader.update(batch_size=4)
val_dataloader.update(batch_size=2)
test_dataloader.update(batch_size=2)

# # debug settings
# train_dataloader.update(dataset=dict(indices=50))
# val_dataloader.update(dataset=dict(indices=50))

# visualization settings
vis_backends = [
    dict(type=LocalVisBackend),
    dict(type=WandbVisBackend, init_kwargs=dict(project="ssc-topk-fuse", name="minkunet-4-lr-nofree")),
]
visualizer = dict(type=OccLocalVisualizer, vis_backends=vis_backends, name="visualizer", ssc_show_dir="outputs/visualizer")

custom_imports = dict(imports=["projects.occ_paper.mmdet3d_plugin"], allow_failed_imports=False)
