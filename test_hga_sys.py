# test_hga_system.py

import torch
from modules import (
    DenseNetVisualEncoder,
    ElectraTextEncoder,
    HGAFusion,  # <--- 测试 HGA
    CrisisKANClassifier,
    ModularCrisisModel
)


def test_hga_pipeline():
    print("\n" + "=" * 40)
    print("🚀 Testing HGA-Net Full Pipeline...")
    print("=" * 40)

    # 1. 配置
    BATCH_SIZE = 2
    NUM_CLASSES = 8

    # 2. 实例化组件 (不加载真实权重以加快速度)
    print("1. Building Modules...")
    vis_enc = DenseNetVisualEncoder(weights_path=None)
    txt_enc = ElectraTextEncoder()  # 自动下载或加载

    fusion = HGAFusion(
        visual_dim=vis_enc.output_dim,  # 1920
        text_dim=txt_enc.output_dim,  # 768
        embed_dim=256
    )

    cls_head = CrisisKANClassifier(
        input_dim=fusion.output_dim,  # 512
        num_classes=NUM_CLASSES
    )

    # 3. 组装
    model = ModularCrisisModel(vis_enc, txt_enc, fusion, cls_head)

    # 4. 假数据
    fake_img = torch.randn(BATCH_SIZE, 3, 224, 224)
    fake_txt = {
        'input_ids': torch.randint(0, 1000, (BATCH_SIZE, 50)),
        'attention_mask': torch.ones(BATCH_SIZE, 50)
    }
    inputs = {'image': fake_img, 'text_tokens': fake_txt}

    # 5. 前向传播
    print("2. Running Forward Pass...")
    logits = model(inputs)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
    print("✅ HGA-Net Pipeline Test Passed!")


if __name__ == "__main__":
    test_hga_pipeline()