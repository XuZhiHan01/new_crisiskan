# test_fusion.py

import torch
from modules import CrisisKANFusion


def test_fusion_module():
    print("\n" + "=" * 40)
    print("🧪 Testing CrisisKANFusion Module...")
    print("=" * 40)

    # 1. 定义维度参数
    B_SIZE = 4
    VIS_DIM = 1920  # 模拟 DenseNet 输出
    TXT_DIM = 768  # 模拟 Electra 输出
    PROJ_DIM = 100

    # 2. 初始化模块
    fusion = CrisisKANFusion(visual_dim=VIS_DIM, text_dim=TXT_DIM, proj_dim=PROJ_DIM)
    print(f"Initialized Fusion Module. Output Dim should be: {PROJ_DIM * 2}")

    # 3. 创建假数据
    fake_img_feat = torch.randn(B_SIZE, VIS_DIM)
    fake_txt_feat = torch.randn(B_SIZE, TXT_DIM)

    # 4. 前向传播
    output = fusion(fake_img_feat, fake_txt_feat)

    # 5. 验证结果
    print(f"Input Visual Shape: {fake_img_feat.shape}")
    print(f"Input Text Shape:   {fake_txt_feat.shape}")
    print(f"Output Shape:       {output.shape}")

    expected_shape = (B_SIZE, PROJ_DIM * 2)
    assert output.shape == expected_shape, f"❌ Shape mismatch! Got {output.shape}, expected {expected_shape}"

    print("✅ Fusion Module Test Passed!")


if __name__ == "__main__":
    test_fusion_module()