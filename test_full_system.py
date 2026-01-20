# test_full_system.py

import torch
from modules import (
    DenseNetVisualEncoder,
    ElectraTextEncoder,
    CrisisKANFusion,
    CrisisKANClassifier,
    ModularCrisisModel
)


def test_full_model():
    print("\n" + "=" * 40)
    print("🚀 Testing Full Modular System...")
    print("=" * 40)

    # 1. 定义配置
    BATCH_SIZE = 2
    NUM_CLASSES = 8  # Task 2 Full

    # 2. 实例化所有子模块 (积木)
    print("1. Building Modules...")
    # 注意：这里我们为了测试速度，VisualEncoder 使用随机初始化 (不传路径或传None)
    # 实际训练时请传真实路径 '../local_models/...'
    vis_enc = DenseNetVisualEncoder(weights_path=None)
    txt_enc = ElectraTextEncoder()  # 会尝试加载本地或自动下载

    fusion = CrisisKANFusion(
        visual_dim=vis_enc.output_dim,
        text_dim=txt_enc.output_dim,
        proj_dim=100
    )

    cls_head = CrisisKANClassifier(
        input_dim=fusion.output_dim,
        num_classes=NUM_CLASSES
    )

    # 3. 组装 (总装)
    print("2. Assembling Model...")
    model = ModularCrisisModel(vis_enc, txt_enc, fusion, cls_head)

    # 4. 准备假数据
    print("3. Generating Fake Data...")
    fake_image = torch.randn(BATCH_SIZE, 3, 224, 224)
    fake_text = {
        'input_ids': torch.randint(0, 1000, (BATCH_SIZE, 50)),
        'attention_mask': torch.ones(BATCH_SIZE, 50)
    }

    # 5. 前向传播
    print("4. Running Forward Pass...")
    # 模拟 Dataset 输出的字典格式
    inputs = {'image': fake_image, 'text_tokens': fake_text}

    logits = model(inputs)

    # 6. 验证
    print(f"Final Logits Shape: {logits.shape}")
    expected_shape = (BATCH_SIZE, NUM_CLASSES)

    assert logits.shape == expected_shape, f"❌ Shape mismatch! Got {logits.shape}"
    print("✅ Full System Test Passed! Ready for training.")


if __name__ == "__main__":
    test_full_model()