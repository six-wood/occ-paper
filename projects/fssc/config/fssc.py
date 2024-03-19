from mmengine.config import read_base
from mmengine.model import MMDistributedDataParallel
from mmengine.visualization.vis_backend import WandbVisBackend, LocalVisBackend
from mmdet3d.visualization import Det3DLocalVisualizer

with read_base():
    from .base.lr import *
    from .base.net import *
    from .base.semankitti import *
    from mmdet3d.configs._base_.default_runtime import *

custom_imports = dict(imports=["projects.fssc.plugin"], allow_failed_imports=False)

default_hooks.update(
    dict(
        logger=dict(type=LoggerHook, interval=10),
        checkpoint=dict(type=CheckpointHook, interval=1),
    )
)


# debug
# train_dataloader.update(batch_size=4)
# val_dataloader.update(batch_size=4)
# train_dataloader.update(dataset=dict(indices=2))
# val_dataloader.update(dataset=dict(indices=2))

# vis_backends = [
#     dict(type=LocalVisBackend),
#     dict(type=WandbVisBackend, init_kwargs=dict(project="pretrain-sc-range", name="minkunet")),
# ]

# visualizer = dict(type=Det3DLocalVisualizer, vis_backends=vis_backends, name="visualizer")
