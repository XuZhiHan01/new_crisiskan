# modules/fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 定义基类
# ==========================================

class BaseFusionModule(nn.Module):
    def __init__(self, visual_dim, text_dim):
        super().__init__()
        self.output_dim = 0

    def forward(self, visual_feats, text_feats):
        raise NotImplementedError


# ==========================================
# 2. 辅助组件：Attention Pooling
# ==========================================
class AttentionPooling(nn.Module):
    """
    自适应地聚合序列特征：
    不是简单的求平均，而是让模型自己决定哪些 token/patch 更重要。
    """

    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        # weights: (Batch, Seq_Len, 1)
        w = self.attention(x)
        # Weighted Sum -> (Batch, Dim)
        return torch.sum(x * w, dim=1)


# ==========================================
# 3. 核心：HGA 融合模块
# ==========================================

class HGAFusion(BaseFusionModule):
    def __init__(self, visual_dim, text_dim, embed_dim=256, num_heads=4, layers=1):
        """
        Args:
            embed_dim: 映射后的统一维度 (如 256)
            num_heads: Transformer 的头数
            layers: Transformer 的层数 (通常 1-2 层足矣)
        """
        super().__init__(visual_dim, text_dim)

        # 1. 维度对齐投影
        # 将不同维度的特征 (1920, 768) 统一映射到 embed_dim
        self.vis_proj = nn.Linear(visual_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        # 2. 双向交互层 (Bi-Directional Cross Attention)
        # 使用 PyTorch 的 TransformerDecoderLayer 来实现 Cross Attention
        # 这里的 "Decoder" 仅仅是指它有 Query 和 Key/Value 的区分

        # A. 图像查询文本 (Image Query, Text Key/Value)
        # 目的：用文本信息来增强图像特征 (比如：根据 "flood" 去高亮图像里的水)
        img_decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.img2text_layers = nn.TransformerDecoder(img_decoder_layer, num_layers=layers)

        # B. 文本查询图像 (Text Query, Image Key/Value)
        # 目的：用图像信息来增强文本特征 (比如：根据图像里的废墟去理解 "collapsed")
        txt_decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.text2img_layers = nn.TransformerDecoder(txt_decoder_layer, num_layers=layers)

        # 3. 智能聚合层 (Attention Pooling)
        # 将交互后的序列 (Batch, Seq, Dim) 聚合成全局向量 (Batch, Dim)
        self.vis_pooling = AttentionPooling(embed_dim)
        self.txt_pooling = AttentionPooling(embed_dim)

        # 4. 最终输出维度
        self.output_dim = embed_dim * 2  # 拼接 (Visual + Text)

    def forward(self, visual_feats, text_feats):
        # visual_feats: (B, 49, 1920)
        # text_feats:   (B, Seq, 768)

        # 1. 投影
        v_embed = self.vis_proj(visual_feats)  # (B, 49, 256)
        t_embed = self.text_proj(text_feats)  # (B, Seq, 256)

        # 2. 深度交互 (Cross Attention)
        # Image looks at Text
        # Tgt=Image, Memory=Text
        v_fused = self.img2text_layers(v_embed, t_embed)  # (B, 49, 256)

        # Text looks at Image
        # Tgt=Text, Memory=Image
        t_fused = self.text2img_layers(t_embed, v_embed)  # (B, Seq, 256)

        # 3. 聚合 (Pooling)
        v_out = self.vis_pooling(v_fused)  # (B, 256)
        t_out = self.txt_pooling(t_fused)  # (B, 256)

        # 4. 拼接
        out = torch.cat([v_out, t_out], dim=1)  # (B, 512)

        return out