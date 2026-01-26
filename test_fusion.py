# test_fusion.py

import torch
from modules import HGAFusion


def test_fusion_module():
    print("\n" + "=" * 40)
    print("🧪 Testing HGA Fusion Module...")
    print("=" * 40)

    # 1. 定义维度参数
    B_SIZE = 4
    VIS_DIM = 1920
    TXT_DIM = 768
    EMBED_DIM = 256

    # 2. 初始化模块
    fusion = HGAFusion(visual_dim=VIS_DIM, text_dim=TXT_DIM, embed_dim=EMBED_DIM)
    expected_out_dim = EMBED_DIM * 2
    print(f"Initialized HGA Fusion. Expected Output Dim: {expected_out_dim}")

    # 3. 创建假数据 (序列形式！)
    # Visual: (Batch, 49, 1920)
    fake_img_feat = torch.randn(B_SIZE, 49, VIS_DIM)
    # Text: (Batch, 50, 768)
    fake_txt_feat = torch.randn(B_SIZE, 50, TXT_DIM)

    # 4. 前向传播
    output = fusion(fake_img_feat, fake_txt_feat)

    # 5. 验证结果
    print(f"Input Visual Shape: {fake_img_feat.shape}")
    print(f"Input Text Shape:   {fake_txt_feat.shape}")
    print(f"Output Shape:       {output.shape}")

    expected_shape = (B_SIZE, expected_out_dim)
    assert output.shape == expected_shape, f"❌ Shape mismatch! Got {output.shape}, expected {expected_shape}"

    print("✅ HGA Fusion Module Test Passed!")


if __name__ == "__main__":
    test_fusion_module()