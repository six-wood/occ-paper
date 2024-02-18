from mmengine.config import read_base
from mmengine.visualization.vis_backend import WandbVisBackend, LocalVisBackend
from projects.occ_paper.mmdet3d_plugin.visualization.local_visualizer import OccLocalVisualizer


with read_base():
    from mmdet3d.configs._base_.default_runtime import *
    from .base.cosine import *
    from .base.net import *
    from .base.semankitti import *

# TODO implement(copy) the focal loss in head
# TODO Add image
# TODO Add image RandomFlip3D
# TODO Use the officail implementation of the fuse layer(MVXTwoStageDetector)
# TODO Change the ssc head to RPN Detection Head(MVXTwoStageDetector,despite the semantic ad geometric)

auto_scale_lr.update(base_batch_size=4, enable=True)

# train_dataloader.update(dataset=dict(indices=50))
# val_dataloader.update(dataset=dict(indices=50))

# train settings
train_dataloader.update(batch_size=4)
val_dataloader.update(batch_size=4)
test_dataloader.update(batch_size=4)

# visualization settings
vis_backends = [
    dict(type=LocalVisBackend),
    dict(type=WandbVisBackend, init_kwargs=dict(project="ssc-lidar", name="Ifree-amp-new-label-mask-range(resnet)-bev")),
]
visualizer = dict(type=OccLocalVisualizer, vis_backends=vis_backends, name="visualizer", ssc_show_dir="outputs/visualizer")

# checkpoint_config = None
default_hooks.update(checkpoint=dict(type=CheckpointHook, interval=1))
train_cfg.update(max_epochs=16)
randomness = dict(seed=3407)
# compile = dict(mode="reduce-overhead")
