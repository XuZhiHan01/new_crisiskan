# modules/__init__.py
from .encoders import DenseNetVisualEncoder, ElectraTextEncoder
from .fusion import HGAFusion, BaseFusionModule
from .classifier import CrisisKANClassifier  # <--- 新增
from .model import ModularCrisisModel