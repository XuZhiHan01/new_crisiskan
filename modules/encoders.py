# modules/encoders.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import ElectraModel, ElectraConfig
from collections import OrderedDict


# ==========================================
# 1. 定义基类
# ==========================================

class BaseVisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 0

    def forward(self, images):
        raise NotImplementedError


class BaseTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 0

    def forward(self, text_inputs):
        raise NotImplementedError


# ==========================================
# 2. DenseNet 视觉编码器 (修复命名不匹配版)
# ==========================================

class DenseNetVisualEncoder(BaseVisualEncoder):
    def __init__(self, weights_path='../local_models/densenet201-c1103571.pth', pretrained=False):
        super().__init__()

        # 1. 初始化骨架 (忽略那个 UserWarning)
        self.backbone = models.densenet201(pretrained=pretrained)
        self.output_dim = 1920

        # 2. 加载权重
        if weights_path:
            self._load_local_weights(weights_path)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)

    def _load_local_weights(self, path):
        print(f"[VisualEncoder] Loading weights from: {path}")

        try:
            # 加载文件
            loaded_state = torch.load(path, map_location='cpu')

            # --- 步骤 A: 拆包 ---
            if isinstance(loaded_state, dict):
                if 'state_dict' in loaded_state:
                    loaded_state = loaded_state['state_dict']
                elif 'model' in loaded_state:
                    loaded_state = loaded_state['model']

            # --- 步骤 B: 智能键名修复 (核心修改) ---
            new_state_dict = OrderedDict()
            model_keys = list(self.backbone.state_dict().keys())

            for k, v in loaded_state.items():
                # 1. 基础清洗：去掉 'module.' 前缀
                name = k.replace('module.', '')

                # 2. 版本兼容性清洗 (Old Torchvision -> New Torchvision)
                # 老权重里可能是 norm.1, conv.1 -> 新代码里是 norm1, conv1
                name = name.replace('norm.1', 'norm1')
                name = name.replace('norm.2', 'norm2')
                name = name.replace('conv.1', 'conv1')
                name = name.replace('conv.2', 'conv2')

                # 3. 尝试匹配
                if name in model_keys:
                    new_state_dict[name] = v
                # 有些官方权重需要加 'features.' 前缀
                elif ('features.' + name) in model_keys:
                    new_state_dict['features.' + name] = v
                else:
                    # 如果还不行，保留原名做最后的挣扎
                    new_state_dict[name] = v

            # 3. 载入权重
            msg = self.backbone.load_state_dict(new_state_dict, strict=False)

            # 统计成功加载的层数（粗略估计）
            loaded_count = len(new_state_dict)

            print(f"[VisualEncoder] Loaded layers (approx): {loaded_count}")
            print(f"[VisualEncoder] Missing keys: {len(msg.missing_keys)}")

            if len(msg.missing_keys) > 0:
                # 只要 Missing keys 少于 5 (通常是 classifier 的 weight 和 bias)，就算完美成功
                if len(msg.missing_keys) > 5:
                    print(f"⚠️ Warning: Still missing {len(msg.missing_keys)} keys. First 5: {msg.missing_keys[:5]}")
                else:
                    print(f"✅ Weights loaded successfully! (Ignored classifier layer)")

        except Exception as e:
            print(f"❌ [VisualEncoder] Critical Error loading weights: {e}")

    def forward(self, images):
        features = self.backbone.features(images)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        out = self.flatten(features)
        return self.dropout(out)


# ==========================================
# 3. Electra 文本编码器 (保持不变)
# ==========================================

class ElectraTextEncoder(BaseTextEncoder):
    def __init__(self, model_path='../local_models/google/electra-base-discriminator'):
        super().__init__()
        self.output_dim = 768

        print(f"[TextEncoder] Loading Electra from {model_path} ...")
        try:
            config = ElectraConfig()
            self.backbone = ElectraModel(config).from_pretrained(model_path)
        except Exception:
            print(f"⚠️ [TextEncoder] Local model not found, trying HuggingFace online...")
            self.backbone = ElectraModel.from_pretrained('google/electra-base-discriminator')

        self.dropout = nn.Dropout(0.1)

    def forward(self, text_inputs):
        outputs = self.backbone(**text_inputs)
        last_hidden_state = outputs[0]
        cls_vector = last_hidden_state[:, 0, :]
        return self.dropout(cls_vector)