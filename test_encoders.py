# test_encoders.py

import torch
import os
from modules import DenseNetVisualEncoder, ElectraTextEncoder


def test_visual_encoder():
    print("\n" + "=" * 40)
    print("🧪 Testing Fine-Grained Visual Encoder...")
    print("=" * 40)

    # 1. 初始化 (使用随机权重测试形状即可)
    encoder = DenseNetVisualEncoder(weights_path=None)

    # 2. 创建假数据 (Batch=2, Channel=3, H=224, W=224)
    fake_image = torch.randn(2, 3, 224, 224)

    # 3. 前向传播
    output = encoder(fake_image)

    # 4. 验证
    print(f"Input Shape: {fake_image.shape}")
    print(f"Output Shape: {output.shape}")

    # 关键修改：现在的期望输出是序列 (Batch, 49, 1920)
    # 49 = 7x7 grid
    expected_shape = (2, 49, 1920)
    assert output.shape == expected_shape, f"❌ Shape mismatch! Expected {expected_shape}, got {output.shape}"
    print("✅ Visual Encoder (Sequence) Test Passed!")


def test_text_encoder():
    print("\n" + "=" * 40)
    print("🧪 Testing Fine-Grained Text Encoder...")
    print("=" * 40)

    # 1. 初始化
    encoder = ElectraTextEncoder()

    # 2. 创建假数据 (Batch=2, SeqLen=50)
    SEQ_LEN = 50
    fake_inputs = {
        'input_ids': torch.randint(0, 1000, (2, SEQ_LEN)),
        'attention_mask': torch.ones(2, SEQ_LEN)
    }

    # 3. 前向传播
    output = encoder(fake_inputs)

    # 4. 验证
    print(f"Input Shape: (Batch=2, Seq={SEQ_LEN})")
    print(f"Output Shape: {output.shape}")

    # 关键修改：现在的期望输出是完整序列 (Batch, 50, 768)
    # 而不是以前的 (Batch, 768)
    expected_shape = (2, SEQ_LEN, 768)
    assert output.shape == expected_shape, f"❌ Shape mismatch! Expected {expected_shape}, got {output.shape}"
    print("✅ Text Encoder (Sequence) Test Passed!")


if __name__ == "__main__":
    test_visual_encoder()
    test_text_encoder()