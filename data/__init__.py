# data/__init__.py

# 暴露核心类，方便外部直接 from data import ...
from .dataset import CrisisDataset
from .text_proc import TextProcessor
from .transforms import get_transforms
from .config import TASK_CONFIG, DEFAULT_DATA_ROOT