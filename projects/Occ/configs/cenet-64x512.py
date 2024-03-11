from mmengine.config import read_base

with read_base():
    from .base.lr import *
    from .base.net import *
    from .base.semankitti import *
    from mmdet3d.configs._base_.default_runtime import *

custom_imports = dict(imports=["projects.Occ.plugin"], allow_failed_imports=False)

model.update(
    dict(
        backbone=dict(
            init_cfg=dict(
                type="Pretrained",
                checkpoint="/home/lms/code/occ-paper/work_dirs/cenet/epoch_45.pth",
                prefix="backbone",
            ),
        ),
        decode_head=dict(
            init_cfg=dict(
                type="Pretrained",
                checkpoint="/home/lms/code/occ-paper/work_dirs/cenet/epoch_45.pth",
                prefix="decode_head",
            ),
        ),
        auxiliary_head=[
            dict(
                type=RangeImageHead,
                channels=128,
                num_classes=20,
                dropout_ratio=0,
                loss_ce=dict(type=CrossEntropyLoss, use_sigmoid=False, class_weight=None, loss_weight=1.0),
                loss_lovasz=dict(type=LovaszLoss, loss_weight=1.5, reduction="none"),
                loss_boundary=dict(type=BoundaryLoss, loss_weight=1.0),
                conv_seg_kernel_size=1,
                ignore_index=free_index,
                indices=2,
                init_cfg=dict(
                    type="Pretrained",
                    checkpoint="/home/lms/code/occ-paper/work_dirs/cenet/epoch_45.pth",
                    prefix="auxiliary_head.0",
                ),
            ),
            dict(
                type=RangeImageHead,
                channels=128,
                num_classes=20,
                dropout_ratio=0,
                loss_ce=dict(type=CrossEntropyLoss, use_sigmoid=False, class_weight=None, loss_weight=1.0),
                loss_lovasz=dict(type=LovaszLoss, loss_weight=1.5, reduction="none"),
                loss_boundary=dict(type=BoundaryLoss, loss_weight=1.0),
                conv_seg_kernel_size=1,
                ignore_index=free_index,
                indices=3,
                init_cfg=dict(
                    type="Pretrained",
                    checkpoint="/home/lms/code/occ-paper/work_dirs/cenet/epoch_45.pth",
                    prefix="auxiliary_head.1",
                ),
            ),
            dict(
                type=RangeImageHead,
                channels=128,
                num_classes=20,
                dropout_ratio=0,
                loss_ce=dict(type=CrossEntropyLoss, use_sigmoid=False, class_weight=None, loss_weight=1.0),
                loss_lovasz=dict(type=LovaszLoss, loss_weight=1.5, reduction="none"),
                loss_boundary=dict(type=BoundaryLoss, loss_weight=1.0),
                conv_seg_kernel_size=1,
                ignore_index=free_index,
                indices=4,
                init_cfg=dict(
                    type="Pretrained",
                    checkpoint="/home/lms/code/occ-paper/work_dirs/cenet/epoch_45.pth",
                    prefix="auxiliary_head.2",
                ),
            ),
        ],
    )
)


# from mmengine.config import read_base

# with read_base():
#     from .base.lr import *
#     from .base.net import *
#     from .base.semankitti import *
#     from mmdet3d.configs._base_.default_runtime import *

# custom_imports = dict(imports=["projects.Occ.plugin"], allow_failed_imports=False)

# model.update(
#     dict(
#         init_cfg=dict(
#             type="Pretrained",
#             checkpoint="/home/lms/code/occ-paper/work_dirs/cenet/epoch_45.pth",
#         )
#     )
# )
