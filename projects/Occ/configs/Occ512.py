from mmengine.config import read_base

with read_base():
    from .base.lr import *
    from .base.net import *
    from .base.semankitti import *
    from mmdet3d.configs._base_.default_runtime import *

custom_imports = dict(imports=["projects.Occ.plugin"], allow_failed_imports=False)

default_hooks.update(
    dict(
        logger=dict(type=LoggerHook, interval=10),
        checkpoint=dict(type=CheckpointHook, interval=1),
    )
)

model.update(
    dict(
        init_cfg=dict(
            type="Pretrained",
            checkpoint="/home/lms/code/occ-paper/work_dirs/cenet/epoch_45.pth",
        )
    )
)

optim_wrapper.update(
    dict(
        paramwise_cfg=dict(
            custom_keys={
                "backbone": dict(lr=1e-3),
                "decode_head": dict(lr=1e-3),
                "auxiliary_head.0": dict(lr=1e-3),
                "auxiliary_head.1": dict(lr=1e-3),
                "auxiliary_head.2": dict(lr=1e-3),
            }
        )
    )
)
# debug
# train_dataloader.update(dataset=dict(indices=1))
# train_dataloader.update(batch_size=1)

# val_dataloader.update(batch_size=1)
# val_dataloader.update(dataset=dict(indices=1))
