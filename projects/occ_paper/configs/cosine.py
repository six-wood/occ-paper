# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import CosineAnnealingLR, LinearLR
from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop
from torch.optim.adamw import AdamW

# This schedule is mainly used by models with dynamic voxelization
# optimizer
lr = 2e-4  # max learning rate
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(type=LinearLR, start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type=CosineAnnealingLR, begin=0, T_max=40, end=40, by_epoch=True, eta_min=1e-5
    ),
]
# training schedule for 1x
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=24, val_interval=1)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
