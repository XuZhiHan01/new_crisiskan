# modules/classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


class CrisisKANClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.15):
        super().__init__()
        # 🌟 黑科技：创建 5 个不同的 Dropout 层
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(5)])
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # 🌟 5个分支分别推断，最后求平均。代码极短，但泛化能力飙升
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.fc(dropout(x))
            else:
                out += self.fc(dropout(x))
        return out / len(self.dropouts)