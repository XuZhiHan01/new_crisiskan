# modules/classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


class CrisisKANClassifier(BaseClassifier):
    def __init__(self, input_dim, num_classes=2, dropout_rate=0.1):
        """
        Args:
            input_dim: 输入特征维度 (例如 200)
            num_classes: 分类数量 (例如 2 或 8)
        """
        super().__init__(input_dim, num_classes)

        # 1. 额外的全连接层 (模拟原论文的 fc_as_self_attn)
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 2. 最终分类层
        self.cls_layer = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: (B, input_dim)
        feat = self.fc_layers(x)
        logits = self.cls_layer(feat)
        return logits