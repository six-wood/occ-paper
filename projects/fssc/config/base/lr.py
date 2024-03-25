# # Copyright (c) OpenMMLab. All rights reserved.
# from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper, OptimWrapper
# from mmengine.optim.scheduler.lr_scheduler import OneCycleLR
# from torch.optim.adamw import AdamW

# # This schedule is mainly used on Semantickitti dataset in segmentation task
# optim_wrapper = dict(
#     type=AmpOptimWrapper,
#     optimizer=dict(
#         type=AdamW,
#         lr=0.01,
#         betas=(0.9, 0.999),
#         weight_decay=(0.01),
#         eps=0.000005,
#     ),
# )

# param_scheduler = [
#     dict(
#         type=OneCycleLR,
#         total_steps=50,
#         by_epoch=True,
#         eta_max=0.0025,
#         pct_start=0.2,
#         div_factor=25.0,
#         final_div_factor=100.0,
#         convert_to_iter_based=True,
#     )
# ]

# # runtime settings
# train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
# val_cfg = dict()
# test_cfg = dict()

# # Default setting for scaling LR automatically
# #   - `enable` means enable scaling LR automatically
# #       or not by default.
# #   - `base_batch_size` = (4 GPUs) x (4 samples per GPU).
# auto_scale_lr = dict(enable=True, base_batch_size=4)

# randomness = dict(seed=3407)

# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.optim.optimizer.amp_optimizer_wrapper import AmpOptimWrapper
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import CosineAnnealingLR, LinearLR
from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop
from torch.optim.adamw import AdamW

# This schedule is mainly used by models with dynamic voxelization
# optimizer
lr = 1e-3  # max learning rate
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(type=LinearLR, start_factor=0.25, begin=0, end=1, by_epoch=True, convert_to_iter_based=True),
    dict(type=CosineAnnealingLR, begin=0, T_max=20, end=20, eta_min=1e-5, by_epoch=True, convert_to_iter_based=True),
]
# training schedule for 1x
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=20, val_interval=1)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (2 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=4)

randomness = dict(seed=3407)
