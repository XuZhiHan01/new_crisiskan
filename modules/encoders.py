# modules/encoders.py
import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import ElectraModel, ElectraConfig
from collections import OrderedDict
from transformers import AutoModel, AutoConfig, AutoTokenizer
from timm.models import load_checkpoint



# ==========================================
# 1. 定义基类
# ==========================================

class BaseVisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 0

    def forward(self, images):
        """
        输出: Feature Sequence (B, N, D)
        """
        raise NotImplementedError


class BaseTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 0

    def forward(self, text_inputs):
        """
        输出: Feature Sequence (B, L, D)
        """
        raise NotImplementedError


# ==========================================
# 2. DenseNet 视觉编码器 (细粒度版)
# ==========================================

class DenseNetVisualEncoder(BaseVisualEncoder):
    def __init__(self, weights_path='../local_models/densenet201-c1103571.pth', pretrained=False):
        super().__init__()

        # 1. 初始化骨架
        self.backbone = models.densenet201(pretrained=pretrained)
        self.output_dim = 1920

        # 2. 加载权重 (复用之前的智能加载逻辑)
        if weights_path:
            self._load_local_weights(weights_path)

        self.dropout = nn.Dropout(0.1)

    def _load_local_weights(self, path):
        print(f"[VisualEncoder] Loading weights from: {path}")
        try:
            loaded_state = torch.load(path, map_location='cpu')
            if isinstance(loaded_state, dict):
                if 'state_dict' in loaded_state:
                    loaded_state = loaded_state['state_dict']
                elif 'model' in loaded_state:
                    loaded_state = loaded_state['model']

            new_state_dict = OrderedDict()
            model_keys = list(self.backbone.state_dict().keys())

            for k, v in loaded_state.items():
                name = k.replace('module.', '')
                # 兼容旧版本权重命名
                name = name.replace('norm.1', 'norm1').replace('norm.2', 'norm2')
                name = name.replace('conv.1', 'conv1').replace('conv.2', 'conv2')

                if name in model_keys:
                    new_state_dict[name] = v
                elif ('features.' + name) in model_keys:
                    new_state_dict['features.' + name] = v
                else:
                    new_state_dict[name] = v

            msg = self.backbone.load_state_dict(new_state_dict, strict=False)
            print(f"[VisualEncoder] Missing keys: {len(msg.missing_keys)}")

        except Exception as e:
            print(f"❌ [VisualEncoder] Error: {e}")

    def forward(self, images):
        # 1. 提取特征图
        # 输入: (B, 3, 224, 224)
        # 输出: (B, 1920, 7, 7)
        features = self.backbone.features(images)
        features = F.relu(features, inplace=True)

        # 2. 空间展平 (Spatial Flattening)
        # (B, 1920, 7, 7) -> (B, 1920, 49)
        B, C, H, W = features.shape
        features = features.view(B, C, H * W)

        # 3. 维度置换 (Permute) -> 适应 Transformer 输入 (Batch, Seq, Dim)
        # (B, 1920, 49) -> (B, 49, 1920)
        features = features.permute(0, 2, 1)

        return self.dropout(features)


# ==========================================
# 3. Electra 文本编码器 (细粒度版)
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
            self.backbone = ElectraModel.from_pretrained('google/electra-base-discriminator')

        self.dropout = nn.Dropout(0.1)

    def forward(self, text_inputs):
        # text_inputs: {input_ids, attention_mask, ...}
        outputs = self.backbone(**text_inputs)

        # 关键修改：返回所有 Token 的序列，不仅仅是 CLS
        # (B, Seq_Len, 768)
        last_hidden_state = outputs.last_hidden_state

        return self.dropout(last_hidden_state)


class ResNetVisualEncoder(BaseVisualEncoder):
    def __init__(self, weights_path='', pretrained=None):
        super().__init__()
        # 加载预训练的 ResNet50
        # 注意：torchvision 新版本推荐用 weights=... 参数
        self.backbone = models.resnet50(weights=None)

        # 2. 手动加载本地权重
        if weights_path:
            print(f"[VisualEncoder] Loading ResNet weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict)
        elif pretrained:
            # 如果没给路径但要求 pretrained，尝试官方自动加载 (需要联网)
            from torchvision.models import ResNet50_Weights
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # 3. 去掉最后两层 (FC 和 AvgPool)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2])
        self.output_dim = 2048
        self.dropout = nn.Dropout(0.1)

    def forward(self, images):
        # 1. 提取特征图 (B, 2048, 7, 7)
        features = self.feature_extractor(images)

        # 2. 空间展平 (B, 2048, 49)
        B, C, H, W = features.shape
        features = features.view(B, C, H * W)

        # 3. 维度置换 -> (B, 49, 2048) 适配 HGA
        features = features.permute(0, 2, 1)

        return self.dropout(features)


# === 新增：BERTweet 文本编码器 ===
class BERTweetTextEncoder(BaseTextEncoder):
    def __init__(self, model_path=''):
        super().__init__()
        self.output_dim = 768

        print(f"[TextEncoder] Loading BERTweet from {model_path} ...")
        # 自动加载配置和模型
        self.backbone = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_inputs):
        outputs = self.backbone(**text_inputs)
        # 获取序列特征 (B, Seq_Len, 768)
        last_hidden_state = outputs.last_hidden_state
        return self.dropout(last_hidden_state)



# ==========================================
# 🚀 现代化视觉编码器: ConvNeXt-V2 (完全纯本地离线版)
# ==========================================
class ConvNextVisualEncoder(nn.Module):
    def __init__(self, weights_path='../local_models/convnextv2_base/model.safetensors'):
        super().__init__()

        # 1. 创建模型结构，去掉分类头 (num_classes=0)
        # 注意：这里去掉了 checkpoint_path 参数
        self.backbone = timm.create_model(
            'convnextv2_base.fcmae_ft_in22k_in1k',
            pretrained=False,
            num_classes=0
        )

        # 2. 手动加载本地权重，并设置 strict=False 忽略多余的分类头
        if weights_path and os.path.exists(weights_path):
            print(f"📥 Loading local visual weights from {weights_path}")
            load_checkpoint(self.backbone, weights_path, strict=False)
        else:
            print("⚠️ Warning: 找不到本地视觉权重，将使用随机初始化！请检查路径。")

        self.output_dim = self.backbone.num_features  # 通常是 1024

    def forward(self, x):
        # 提取空间特征图 (B, 1024, 7, 7)
        features = self.backbone.forward_features(x)
        # 展平空间维度 (B, 49, 1024)，完美对接你的 C-GMAN Fusion
        features = features.flatten(2).transpose(1, 2)
        return features


# 🚀 现代化文本编码器: DeBERTa-V3 (纯本地离线版)
# ==========================================
class DebertaTextEncoder(nn.Module):
    def __init__(self, model_path='../local_models/deberta-v3-base'):
        super().__init__()
        # 直接传入本地文件夹路径
        self.backbone = AutoModel.from_pretrained(model_path)
        self.output_dim = self.backbone.config.hidden_size # 通常是 768

    def forward(self, text_tokens):
        # 🌟 关键修改：接收字典 text_tokens，并从中提取 input_ids 和 attention_mask
        input_ids = text_tokens['input_ids']
        attention_mask = text_tokens.get('attention_mask', None)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # 获取 Token 序列 (B, Seq_Len, 768)
        sequence_output = outputs.last_hidden_state
        return sequence_output