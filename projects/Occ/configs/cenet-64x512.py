from mmengine.config import read_base

with read_base():
    from .base.lr import *
    from .base.net import *
    from .base.semankitti import *
    from mmdet3d.configs._base_.default_runtime import *

custom_imports = dict(imports=["projects.Occ.plugin"], allow_failed_imports=False)
