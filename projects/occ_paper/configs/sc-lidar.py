from mmengine.config import read_base

with read_base():
    from mmdet3d.configs._base_.default_runtime import *
    from .cosine import *
    from .rvnet import *
    from .semankitti import *

# train_dataloader.update(dataset=dict(indices=50))
# val_dataloader.update(dataset=dict(indices=50))

log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])

# checkpoint_config = None
checkpoint_config = dict(interval=1)
