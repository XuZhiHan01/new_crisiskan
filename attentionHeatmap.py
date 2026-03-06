# visualize_attention.py

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 导入你现有的模块
from data import TextProcessor, get_transforms, TASK_CONFIG
from modules import (
    ResNetVisualEncoder,
    BERTweetTextEncoder,
    CGMANFusion,
    CrisisKANClassifier,
    ModularCrisisModel
)


def build_cgman_model(device, num_classes):
    """重建 C-GMAN 模型并加载权重"""
    vis_enc = ResNetVisualEncoder(weights_path=None, pretrained=False)
    txt_enc = BERTweetTextEncoder(model_path='../local_models/vinai/bertweet-base')

    fusion = CGMANFusion(
        visual_dim=vis_enc.output_dim,
        text_dim=txt_enc.output_dim,
        embed_dim=256,
        num_heads=4,
        layers=1
    )

    cls_head = CrisisKANClassifier(input_dim=fusion.output_dim, num_classes=num_classes)
    model = ModularCrisisModel(vis_enc, txt_enc, fusion, cls_head)

    # 替换为你实际训练出的权重路径
    ckpt_path = './output_cgman/cgman_resnet_bertweet_exp/best_model.pt'
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("✅ Weights Loaded!")
    else:
        print("⚠️ Warning: Checkpoint not found, using random weights for demonstration.")

    model.to(device)
    model.eval()
    return model


def visualize_attention(image_path, text, model, text_processor, device, output_path):
    # 1. 准备数据
    transform = get_transforms(mode='eval')
    raw_img = Image.open(image_path).convert('RGB')
    img_tensor = transform(raw_img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    text_tokens = text_processor(text)
    text_tokens = {k: v.unsqueeze(0).to(device) for k, v in text_tokens.items()}

    inputs = {'image': img_tensor, 'text_tokens': text_tokens}

    # 2. 注册 Hook，优雅提取内部交互特征 (完美规避黑盒问题)
    features = {}

    def hook_v(module, input, output): features['v_intra'] = output

    def hook_t(module, input, output): features['t_intra'] = output

    # 挂载到自注意力输出层
    handle_v = model.fusion_module.vis_self_attn.register_forward_hook(hook_v)
    handle_t = model.fusion_module.txt_self_attn.register_forward_hook(hook_t)

    # 3. 前向推理
    with torch.no_grad():
        logits = model(inputs)
        pred_id = torch.argmax(logits, dim=1).item()

    # 卸载 Hook
    handle_v.remove()
    handle_t.remove()

    # 4. 手动计算 Text-to-Image 的跨模态注意力分布
    v_intra = features['v_intra']  # (1, 49, 256)
    t_intra = features['t_intra']  # (1, Seq, 256)

    # 用文本的全局 [CLS] Token (索引0) 去 Query 图片的 49 个网格
    text_cls = t_intra[:, 0:1, :]  # (1, 1, 256)
    attn_weights = torch.bmm(text_cls, v_intra.transpose(1, 2))  # (1, 1, 49)

    # 缩放点积注意力公式计算概率
    attn_weights = torch.softmax(attn_weights / np.sqrt(256), dim=-1)

    # 将 (49,) reshape 成 (7, 7) 的空间热力图
    attn_map = attn_weights.squeeze().cpu().numpy()
    attn_map = attn_map.reshape(7, 7)

    # 5. 生成可视化图像
    img_bgr = cv2.imread(image_path)
    img_bgr = cv2.resize(img_bgr, (224, 224))

    # 归一化并上采样到 224x224
    attn_map_resized = cv2.resize(attn_map, (224, 224), interpolation=cv2.INTER_CUBIC)
    attn_map_norm = np.uint8(255 * (attn_map_resized - attn_map_resized.min()) /
                             (attn_map_resized.max() - attn_map_resized.min() + 1e-8))

    # 映射为伪彩色热力图 (红高蓝低)
    heatmap = cv2.applyColorMap(attn_map_norm, cv2.COLORMAP_JET)

    # 叠加 (0.6为原图权重，0.4为热力图权重)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

    # 6. 绘图并保存
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Attention Heatmap (Pred Class: {pred_id})")
    plt.axis('off')

    # 在底部显示推文内容
    import textwrap
    wrapped_text = textwrap.fill(text, width=80)
    plt.suptitle(f"Tweet: {wrapped_text}", fontsize=10, y=0.05)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化结果已保存至: {output_path}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_processor = TextProcessor(model_name='../local_models/vinai/bertweet-base')
    model = build_cgman_model(device, num_classes=8)  # Task 2 是 8 个类别

    # ================= 替换为你真实的测试图片和推文 =================
    test_image_path = "/home/tSdu/xzh/crisisKAN/crisiskan/datasets/settingA/data_image/california_wildfires/10_10_2017/917793137925459968_0.jpg"  # 换成你硬盘里存在的图
    test_text = "California wildfires destroy more than 50 structures"
    output_img_path = "./attention_demo.png"

    if os.path.exists(test_image_path):
        visualize_attention(test_image_path, test_text, model, text_processor, device, output_img_path)
    else:
        print(f"❌ 找不到图片: {test_image_path}，请修改路径后再试。")