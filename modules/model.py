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
        全模块化组装模型 (已适配 C-GMAN 对比学习特征)
        """
        super().__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.fusion_module = fusion_module
        self.classifier = classifier

    def forward(self, inputs, return_features=False):
        """
        Args:
            inputs: 一个包含 'image' 和 'text_tokens' 的字典/元组
            return_features (bool): 如果为 True，则返回全局图文特征用于计算对比损失
        """
        # 1. 解包数据
        if isinstance(inputs, dict):
            image = inputs['image']
            text_tokens = inputs['text_tokens']
        else:
            image, text_tokens = inputs

        # 2. 独立编码 (Encoders)
        v_feat = self.visual_encoder(image)
        t_feat = self.text_encoder(text_tokens)

        # 3. 融合 (Fusion)
        fusion_out = self.fusion_module(v_feat, t_feat)

        # 【关键修改】：兼容性处理
        # 如果是 CGMANFusion，它会返回 (fused_feat, v_global, t_global)
        # 如果是旧版 Fusion，它只返回 fused_feat
        if isinstance(fusion_out, tuple) and len(fusion_out) == 3:
            fused_feat, v_global, t_global = fusion_out
        else:
            fused_feat = fusion_out
            v_global, t_global = None, None

        # 4. 分类 (Classifier)
        logits = self.classifier(fused_feat)

        # 5. 根据需求返回
        if return_features:
            return logits, v_global, t_global
        else:
            return logits