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


class CGMANFusion(BaseFusionModule):
    def __init__(self, visual_dim, text_dim, embed_dim=256, num_heads=4, layers=1):
        """
        C-GMAN: Contrastive-Guided Gated Multimodal Attention Network
        """
        super().__init__(visual_dim, text_dim)

        # 1. 维度对齐投影
        self.vis_proj = nn.Linear(visual_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        # 2. 阶段一：模态内自我增强 (Intra-modal Self-Attention)
        encoder_layer_v = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.vis_self_attn = nn.TransformerEncoder(encoder_layer_v, num_layers=layers)

        encoder_layer_t = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.txt_self_attn = nn.TransformerEncoder(encoder_layer_t, num_layers=layers)

        # 3. 阶段二：跨模态双向注意力 (Inter-modal Cross-Attention)
        # Image queries Text
        decoder_layer_img2txt = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.img2text_cross_attn = nn.TransformerDecoder(decoder_layer_img2txt, num_layers=layers)

        # Text queries Image
        decoder_layer_txt2img = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.text2img_cross_attn = nn.TransformerDecoder(decoder_layer_txt2img, num_layers=layers)

        # 4. 智能聚合层 (复用之前的 AttentionPooling)
        self.vis_pooling = AttentionPooling(embed_dim)
        self.txt_pooling = AttentionPooling(embed_dim)

        # 5. 动态模态感知门控 (Dynamic Modality-Aware Gating)
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),  # 输出两个权重: alpha(文本), beta(视觉)
            nn.Sigmoid()
        )

        # 最终输出维度 (注意：因为采用了加权相加，维度是 embed_dim，而不是 HGA 的 embed_dim * 2)
        self.output_dim = embed_dim

    def forward(self, visual_feats, text_feats):
        # visual_feats: (B, 49, 1920 or 2048)
        # text_feats:   (B, Seq_len, 768)

        # 1. 投影到统一维度
        v_embed = self.vis_proj(visual_feats)  # (B, 49, 256)
        t_embed = self.text_proj(text_feats)  # (B, Seq, 256)

        # [关键前置操作] 提取出全局特征，留给外面计算 对比损失(Contrastive Loss) 用
        v_global = v_embed.mean(dim=1)  # 图像全局均值池化 (B, 256)
        t_global = t_embed[:, 0, :]  # 文本取 [CLS] token (B, 256)

        # 2. 模态内自注意力
        v_intra = self.vis_self_attn(v_embed)  # (B, 49, 256)
        t_intra = self.txt_self_attn(t_embed)  # (B, Seq, 256)

        #v_fused = self.img2text_cross_attn(tgt=v_embed, memory=t_embed)  # (B, 49, 256)
        # Tgt=Text, Memory=Image -> Text 寻找相关的 Image
        #t_fused = self.text2img_cross_attn(tgt=t_embed, memory=v_embed)
        # 3. 跨模态双向注意力
        # Tgt=Image, Memory=Text -> Image 寻找相关的 Text
        v_fused = self.img2text_cross_attn(tgt=v_intra, memory=t_intra)  # (B, 49, 256)
        # Tgt=Text, Memory=Image -> Text 寻找相关的 Image
        t_fused = self.text2img_cross_attn(tgt=t_intra, memory=v_intra)  # (B, Seq, 256)

        # 【消融跨模态】：直接把模态内自注意力的结果送去 Pooling
        #v_fused = v_intra
        #t_fused = t_intra

        # 4. 序列池化降维
        v_pool = self.vis_pooling(v_fused)  # (B, 256)
        t_pool = self.txt_pooling(t_fused)  # (B, 256)

        # 5. 动态门控融合计算权重
        concat_feat = torch.cat([v_pool, t_pool], dim=1)  # (B, 512)
        gates = self.gate_network(concat_feat)  # (B, 2)

        alpha = gates[:, 0:1]  # 文本模态的门控权重 (B, 1)
        beta = gates[:, 1:2]  # 视觉模态的门控权重 (B, 1)

        # 最终门控融合：动态选择更值得信任的模态
        final_fused_feat = alpha * t_pool + beta * v_pool  # (B, 256)
        #final_fused_feat = (t_pool + v_pool) / 2.0  # (B, 256)
        # 返回融合后的特征，以及全局特征(用于计算辅助Loss)
        return final_fused_feat, v_global, t_global