# modules/model.py

import torch
import torch.nn as nn


class ModularCrisisModel(nn.Module):
    def __init__(self,
                 visual_encoder,
                 text_encoder,
                 fusion_module,
                 classifier):
        """
        全模块化组装模型
        """
        super().__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.fusion_module = fusion_module
        self.classifier = classifier

    def forward(self, inputs):
        """
        Args:
            inputs: 一个包含 'image' 和 'text_tokens' 的字典/元组
                    (适配 Dataset 的输出格式)
        """
        # 1. 解包数据
        # 假设 inputs 是从 DataLoader 出来的字典
        if isinstance(inputs, dict):
            image = inputs['image']
            text_tokens = inputs['text_tokens']
        else:
            # 兼容旧代码的元组格式 (image, text)
            image, text_tokens = inputs

        # 2. 独立编码 (Encoders)
        #    Image -> (B, 1920)
        v_feat = self.visual_encoder(image)
        #    Text -> (B, 768)
        t_feat = self.text_encoder(text_tokens)

        # 3. 融合 (Fusion)
        #    (B, 1920) + (B, 768) -> (B, 200)
        fused_feat = self.fusion_module(v_feat, t_feat)

        # 4. 分类 (Classifier)
        #    (B, 200) -> (B, Num_Classes)
        logits = self.classifier(fused_feat)

        return logits