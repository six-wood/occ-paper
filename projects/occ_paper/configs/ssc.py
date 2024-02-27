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

# train settings
train_dataloader.update(batch_size=4)
val_dataloader.update(batch_size=4)
test_dataloader.update(batch_size=4)

# # debug settings
train_dataloader.update(dataset=dict(indices=50))
val_dataloader.update(dataset=dict(indices=50))
train_dataloader.update(batch_size=1)
val_dataloader.update(batch_size=1)
test_dataloader.update(batch_size=1)

# visualization settings
vis_backends = [
    dict(type=LocalVisBackend),
    # dict(type=WandbVisBackend, init_kwargs=dict(project="ssc-topk-fuse", name="dense-baseline")),
]
visualizer = dict(type=OccLocalVisualizer, vis_backends=vis_backends, name="visualizer", ssc_show_dir="outputs/visualizer")
