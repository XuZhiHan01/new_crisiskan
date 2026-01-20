# test_encoders.py

import torch
import os
from modules import DenseNetVisualEncoder, ElectraTextEncoder


def test_visual_encoder():
    print("\n" + "=" * 40)
    print("🧪 Testing DenseNetVisualEncoder...")
    print("=" * 40)

    # 1. 模拟一个伪造的权重路径 (或者填写真实的路径来测试加载)
    # 如果你没有把权重文件移过来，可以填 None 测试纯随机初始化的模型
    weight_path = '../local_models/densenet201-c1103571.pth'
    if not os.path.exists(weight_path):
        print(f"⚠️ Warning: Weight file not found at {weight_path}. Testing with random init.")
        weight_path = None

    encoder = DenseNetVisualEncoder(weights_path=weight_path)

    # 2. 创建假数据 (Batch=2, Channel=3, H=224, W=224)
    fake_image = torch.randn(2, 3, 224, 224)

    # 3. 前向传播
    output = encoder(fake_image)

    # 4. 验证
    print(f"Input Shape: {fake_image.shape}")
    print(f"Output Shape: {output.shape}")

    assert output.shape == (2, 1920), f"❌ Shape mismatch! Expected (2, 1920), got {output.shape}"
    print("✅ Visual Encoder Test Passed!")


def test_text_encoder():
    print("\n" + "=" * 40)
    print("🧪 Testing ElectraTextEncoder...")
    print("=" * 40)

    # 1. 初始化
    # 确保你的 Electra 模型路径是正确的，或者让它联网下载
    model_path = '../local_models/google/electra-base-discriminator'
    encoder = ElectraTextEncoder(model_path=model_path)

    # 2. 创建假数据 (Batch=2, SeqLen=50)
    fake_inputs = {
        'input_ids': torch.randint(0, 1000, (2, 50)),
        'attention_mask': torch.ones(2, 50)
    }

    # 3. 前向传播
    output = encoder(fake_inputs)

    # 4. 验证
    print(f"Input Shape: (Batch=2, Seq=50)")
    print(f"Output Shape: {output.shape}")

    assert output.shape == (2, 768), f"❌ Shape mismatch! Expected (2, 768), got {output.shape}"
    print("✅ Text Encoder Test Passed!")


if __name__ == "__main__":
    test_visual_encoder()
    test_text_encoder()