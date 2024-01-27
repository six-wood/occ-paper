from mmengine.config import read_base
from mmengine.visualization.vis_backend import WandbVisBackend, LocalVisBackend
from mmdet3d.visualization.local_visualizer import Det3DLocalVisualizer


with read_base():
    from mmdet3d.configs._base_.default_runtime import *
    from .cosine import *
    from .rvnet import *
    from .semankitti import *

# train_dataloader.update(dataset=dict(indices=50))
# val_dataloader.update(dataset=dict(indices=50))

# visualization settings
vis_backends = [
    dict(type=LocalVisBackend),
    dict(type=WandbVisBackend, init_kwargs=dict(project="occ_paper", name="rvnet-cross-channel-atten-e40")),
]
visualizer = dict(type=Det3DLocalVisualizer, vis_backends=vis_backends, name="visualizer")

# checkpoint_config = None
checkpoint_config = dict(interval=1)
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=40, val_interval=1)