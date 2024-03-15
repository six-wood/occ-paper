from mmengine.config import read_base
from mmengine.model import MMDistributedDataParallel

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
        backbone=dict(
            init_cfg=dict(
                type="Pretrained",
                checkpoint="/home/lms/code/occ-paper/work_dirs/cenet/epoch_45.pth",
                prefix="backbone",
            ),
        ),
        bev_backbone=dict(
            init_cfg=dict(
                type="Pretrained",
                checkpoint="/home/lms/code/occ-paper/work_dirs/sc/epoch_18.pth",
                prefix="bev_backbone",
            ),
        ),
        decode_head=dict(
            init_cfg=dict(
                type="Pretrained",
                checkpoint="/home/lms/code/occ-paper/work_dirs/cenet/epoch_45.pth",
                prefix="decode_head",
            ),
        ),
        sc_head=dict(
            init_cfg=dict(
                type="Pretrained",
                checkpoint="/home/lms/code/occ-paper/work_dirs/sc/epoch_18.pth",
                prefix="sc_head",
            ),
        ),
    )
)

optim_wrapper.update(
    dict(
        paramwise_cfg=dict(
            custom_keys={
                "backbone": dict(lr_mult=0.1),
                "decode_head": dict(lr_mult=0.1),
                "bev_backbone": dict(lr_mult=0.1),
                "sc_head": dict(lr_mult=0.1),
                "sparse_backbone": dict(lr_mult=1),
                "ssc_head": dict(lr_mult=1),
            }
        )
    )
)

model_wrapper_cfg = dict(type="MMDistributedDataParallel", detect_anomalous_params=True)

# debug
# train_dataloader.update(dataset=dict(indices=2))
# train_dataloader.update(batch_size=2)

# val_dataloader.update(batch_size=2)
# val_dataloader.update(dataset=dict(indices=2))
