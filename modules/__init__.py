# modules/__init__.py
from .encoders import (
    DenseNetVisualEncoder,
    ElectraTextEncoder,
    ResNetVisualEncoder,    # <--- 新增
    BERTweetTextEncoder     # <--- 新增
)
from .fusion import HGAFusion, BaseFusionModule, CGMANFusion
from .classifier import CrisisKANClassifier  # <--- 新增
from .model import ModularCrisisModel