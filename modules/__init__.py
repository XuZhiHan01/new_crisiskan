# modules/__init__.py
from .encoders import DenseNetVisualEncoder, ElectraTextEncoder
from .fusion import CrisisKANFusion # <--- 新增这行
from .classifier import CrisisKANClassifier  # <--- 新增
from .model import ModularCrisisModel