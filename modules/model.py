# modules/model.py

import torch
import torch.nn as nn


import torch
import torch.nn as nn

class ModularCrisisModel(nn.Module):
    def __init__(self,
                 visual_encoder,
                 text_encoder,
                 fusion_module,
                 classifier,
                 num_classes=None,   # 🌟 新增：传入类别数
                 embed_dim=256):     # 🌟 新增：传入特征对齐维度
        """
        全模块化组装模型 (已适配 C-GMAN 深层监督/辅助分类机制)
        """
        super().__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.fusion_module = fusion_module
        self.classifier = classifier

        # 🌟 核心创新：模态惰性惩罚分类头 (辅助分类器)
        if num_classes is not None:
            self.aux_vis_head = nn.Linear(embed_dim, num_classes)
            self.aux_txt_head = nn.Linear(embed_dim, num_classes)
        else:
            self.aux_vis_head = None
            self.aux_txt_head = None

    def forward(self, inputs, return_features=False):
        if isinstance(inputs, dict):
            image = inputs['image']
            text_tokens = inputs['text_tokens']
        else:
            image, text_tokens = inputs

        # 独立编码
        v_feat = self.visual_encoder(image)
        t_feat = self.text_encoder(text_tokens)

        # 融合
        fusion_out = self.fusion_module(v_feat, t_feat)

        if isinstance(fusion_out, tuple) and len(fusion_out) == 3:
            fused_feat, v_global, t_global = fusion_out
        else:
            fused_feat = fusion_out
            v_global, t_global = None, None

        # 主分类器预测
        logits = self.classifier(fused_feat)

        if return_features:
            # 🌟 关键修改：如果启用了返回特征，并且有辅助头，则直接计算辅助分类结果
            if self.aux_vis_head is not None and v_global is not None:
                aux_v_logits = self.aux_vis_head(v_global)
                aux_t_logits = self.aux_txt_head(t_global)
                return logits, aux_v_logits, aux_t_logits
            else:
                return logits, v_global, t_global
        else:
            return logits