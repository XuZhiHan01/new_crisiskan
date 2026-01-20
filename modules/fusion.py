# modules/fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 定义基类 (Base Class)
# ==========================================

class BaseFusionModule(nn.Module):
    def __init__(self, visual_dim, text_dim):
        super().__init__()
        self.output_dim = 0  # 必须定义

    def forward(self, visual_feats, text_feats):
        """
        输入:
            visual_feats: (B, V_dim)
            text_feats:   (B, T_dim)
        输出:
            fused_feat:   (B, Output_Dim)
        """
        raise NotImplementedError


# ==========================================
# 2. 实现 CrisisKAN 核心融合层
# ==========================================

class CrisisKANFusion(BaseFusionModule):
    def __init__(self, visual_dim, text_dim, proj_dim=100):
        """
        Args:
            visual_dim: 视觉特征维度 (例如 1920)
            text_dim:   文本特征维度 (例如 768)
            proj_dim:   投影后的维度 (原论文默认为 100)
        """
        super().__init__(visual_dim, text_dim)

        # 定义投影层 (Projection)
        self.proj_visual = nn.Linear(visual_dim, proj_dim)
        self.proj_visual_bn = nn.BatchNorm1d(proj_dim)

        self.proj_text = nn.Linear(text_dim, proj_dim)
        self.proj_text_bn = nn.BatchNorm1d(proj_dim)

        # 定义注意力层 (用于生成 Gate)
        # 这里的输入维度仍然是原始特征维度，输出是 proj_dim
        self.layer_attn_visual = nn.Linear(visual_dim, proj_dim)
        self.layer_attn_text = nn.Linear(text_dim, proj_dim)

        # 最终输出是拼接后的维度
        self.output_dim = proj_dim * 2

    def _batch_self_attention(self, input_tensor):
        """
        Batch Self-Attention: 计算 Batch 内样本间的相似度
        input: (B, D)
        output: (B, D)
        """
        # (B, D) x (D, B) -> (B, B)
        attn_scores = torch.matmul(input_tensor, input_tensor.transpose(-1, -2))

        # Softmax 归一化
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 加权求和 (B, B) x (B, D) -> (B, D)
        attn_output = torch.matmul(attn_weights, input_tensor)
        return attn_output

    def forward(self, f_i, e_i):
        # f_i: 图像特征 (B, 1920)
        # e_i: 文本特征 (B, 768)

        # 1. 应用 Batch Self-Attention (特征增强)
        #    利用同一个 Batch 里其他样本的信息来增强当前样本
        f_i_attn = self._batch_self_attention(f_i)  # (B, 1920)
        e_i_attn = self._batch_self_attention(e_i)  # (B, 768)

        # 2. 投影 (Projection) -> 映射到 100 维
        #    f_i_tilde = ReLU(BN(Linear(f_i_attn)))
        f_i_tilde = self.proj_visual(f_i_attn)
        f_i_tilde = self.proj_visual_bn(f_i_tilde)
        f_i_tilde = F.relu(f_i_tilde)  # (B, 100)

        e_i_tilde = self.proj_text(e_i_attn)
        e_i_tilde = self.proj_text_bn(e_i_tilde)
        e_i_tilde = F.relu(e_i_tilde)  # (B, 100)

        # 3. 交叉门控 (Cross Gating / Attention)
        #    用文本信息生成 Gate 控制图片，反之亦然

        # Gate for Visual: 由 Text 特征生成
        # alpha_v = sigmoid(Linear(e_i_attn))
        alpha_v = torch.sigmoid(self.layer_attn_text(e_i_attn))  # (B, 100)

        # Gate for Text: 由 Visual 特征生成
        # alpha_e = sigmoid(Linear(f_i_attn))
        alpha_e = torch.sigmoid(self.layer_attn_visual(f_i_attn))  # (B, 100)

        # 4. 加权 (Masking)
        masked_v = torch.mul(alpha_v, f_i_tilde)  # (B, 100)
        masked_e = torch.mul(alpha_e, e_i_tilde)  # (B, 100)

        # 5. 拼接 (Concatenation)
        joint_repr = torch.cat((masked_v, masked_e), dim=1)  # (B, 200)

        return joint_repr