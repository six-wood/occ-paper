from mmengine.config import read_base
from mmengine.visualization.vis_backend import WandbVisBackend, LocalVisBackend
from projects.occ_paper.mmdet3d_plugin.visualization.local_visualizer import OccLocalVisualizer


with read_base():
    from mmdet3d.configs._base_.default_runtime import *
    from .base.cosine import *
    from .base.net import *
    from .base.semankitti import *

# TODO implement(copy) the focal loss in head
# TODO Add image RandomFlip3D
# TODO Use the officail implementation of the fuse layer(MVXTwoStageDetector)
# TODO Change the ssc head to RPN Detection Head(MVXTwoStageDetector,despite the semantic ad geometric)
# TODO Change to Top-K

auto_scale_lr.update(base_batch_size=4, enable=True)

# train settings
train_dataloader.update(batch_size=4)
val_dataloader.update(batch_size=4)
test_dataloader.update(batch_size=4)

# train_dataloader.update(dataset=dict(indices=50))
# val_dataloader.update(dataset=dict(indices=50))
# train_dataloader.update(batch_size=1)
# val_dataloader.update(batch_size=1)
# test_dataloader.update(batch_size=1)

# visualization settings
vis_backends = [
    dict(type=LocalVisBackend),
    dict(type=WandbVisBackend, init_kwargs=dict(project="ssc-lidar", name="baseline-100iterwarmup-20CosineAnnealingLR*1e-3")),
]
visualizer = dict(type=OccLocalVisualizer, vis_backends=vis_backends, name="visualizer", ssc_show_dir="outputs/visualizer")

# checkpoint_config = None
lr = 2e-4  # max learning rate
optim_wrapper.update(dict(optimizer=dict(lr=lr * 5)))
default_hooks.update(dict(checkpoint=dict(interval=1)))
param_scheduler = [
    dict(type=LinearLR, start_factor=0.25, by_epoch=False, begin=0, end=100),
    dict(type=CosineAnnealingLR, begin=0, T_max=20, end=20, by_epoch=True, eta_min=1e-5),
]
train_cfg.update(max_epochs=20)
randomness = dict(seed=3407)
# compile = dict(mode="reduce-overhead")

# optimizer
