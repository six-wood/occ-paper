from mmengine.config import read_base

custom_imports = dict(imports=["projects.sc.plugin"], allow_failed_imports=False)
with read_base():
    from .base.lr import *
    from .base.net import *
    from .base.semankitti import *
    from mmdet3d.configs._base_.default_runtime import *


# debug
# train_dataloader.update(dataset=dict(indices=5))
# train_dataloader.update(batch_size=2)

# val_dataloader.update(batch_size=2)
# val_dataloader.update(dataset=dict(indices=5))
