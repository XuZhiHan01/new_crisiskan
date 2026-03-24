import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

# 导入你的现有组件
from data import TextProcessor, get_transforms, TASK_CONFIG
from modules import (
    ConvNextVisualEncoder,
    DebertaTextEncoder,
    CGMANFusion,
    CrisisKANClassifier,
    ModularCrisisModel
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate C-GMAN Cross-Attention Heatmaps")
    parser.add_argument('--task_name', type=str, default='task2')

    parser.add_argument('--model_path', type=str,
                        default='/home/tSdu/xzh/crisisKAN/crisiskan/new_crisiskan/output_cgman/'
                                'task2/task2_lamda_0.1/best_model.pt',
                        help='最优模型的权重路径')

    parser.add_argument('--image_path', type=str,
                        default='/home/tSdu/xzh/crisisKAN/crisiskan/datasets/settingA/data_image/'
                                'california_wildfires/10_10_2017/917793137925459968_0.jpg',
                        help='测试用的灾害图片路径')

    parser.add_argument('--text', type=str,
                        default='California wildfires destroy more than 50 structures',
                        help='图片对应的推文文本')

    parser.add_argument('--text_model_path', type=str, default='../local_models/deberta-v3-base')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='./heatmaps')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("⏳ 正在加载 C-GMAN 模型 (含深层监督机制)...")
    # 1. 实例化预处理与分词器
    text_proc = TextProcessor(model_name=args.text_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    eval_transform = get_transforms(mode='eval')
    num_classes = TASK_CONFIG[args.task_name]['num_classes']

    # 2. 组装模型
    vis_enc = ConvNextVisualEncoder()
    txt_enc = DebertaTextEncoder(model_path=args.text_model_path)
    fusion = CGMANFusion(visual_dim=vis_enc.output_dim, text_dim=txt_enc.output_dim, embed_dim=args.embed_dim)
    cls_head = CrisisKANClassifier(input_dim=fusion.output_dim, num_classes=num_classes)

    # 🌟 必须加上 num_classes，匹配带有防偷懒机制的权重
    model = ModularCrisisModel(vis_enc, txt_enc, fusion, cls_head, num_classes=num_classes, embed_dim=args.embed_dim)

    # 加载权重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    print("🖼️ 正在处理输入图像与文本...")
    # 3. 处理输入数据
    image = Image.open(args.image_path).convert('RGB')
    image_tensor = eval_transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # 这里的 TextProcessor 已经支持直接调用
    text_inputs = text_proc(args.text)
    text_inputs = {k: v.unsqueeze(0).to(device) for k, v in text_inputs.items()}

    # 获取 token 对应的实际单词列表
    tokens = tokenizer.convert_ids_to_tokens(text_inputs['input_ids'][0])

    print("🧠 正在深入模型提取跨模态注意力矩阵...")
    # 4. 提取交叉注意力权重
    with torch.no_grad():
        v_feat = model.visual_encoder(image_tensor)
        t_feat = model.text_encoder(text_inputs)

        v_embed = model.fusion_module.vis_proj(v_feat)
        t_embed = model.fusion_module.text_proj(t_feat)
        v_intra = model.fusion_module.vis_self_attn(v_embed)
        t_intra = model.fusion_module.txt_self_attn(t_embed)

        # 提取 Text2Img 交叉注意力权重
        attn_output, attn_weights = model.fusion_module.text2img_cross_attn.layers[0].multihead_attn(
            query=t_intra,
            key=v_intra,
            value=v_intra,
            need_weights=True
        )
        attn_weights = attn_weights.squeeze(0).cpu().numpy()  # (Seq_Len, 49)

    print(f"🎨 正在叠加生成高清热力图，结果将保存至 {args.output_dir}...")
    # 5. 可视化与保存 (🌟 关键修改：恢复原图的真实高清分辨率)
    orig_img = cv2.imread(args.image_path)
    # 获取原图的真实高度和宽度 (例如 1920x1080)
    real_h, real_w = orig_img.shape[:2]

    # 用来收集所有有效单词的注意力分布，以便最后计算整体均值
    valid_attn_list = []

    # ==========================================
    # 阶段 A：生成每个有效单词的高清热力图
    # ==========================================
    for idx, token in enumerate(tokens):
        clean_token = token.replace('Ġ', '').replace(' ', '').replace('##', '')
        if clean_token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>', '', '.', ',', ';', ':', 'http', 'https']:
            continue

        token_attn_raw = attn_weights[idx]
        valid_attn_list.append(token_attn_raw)

        token_attn = token_attn_raw.reshape(7, 7)

        token_attn = token_attn - np.min(token_attn)
        if np.max(token_attn) > 0:
            token_attn = token_attn / np.max(token_attn)

        # 🌟 关键修改：将 7x7 的热力图直接插值放大到原图的真实尺寸 (real_w, real_h)
        heatmap = cv2.resize(token_attn, (real_w, real_h))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 叠加到高清原图上
        superimposed_img = heatmap_color * 0.4 + orig_img * 0.6

        save_path = os.path.join(args.output_dir, f"attn_word_{idx}_{clean_token}.jpg")
        cv2.imwrite(save_path, superimposed_img)
        print(f"   ✅ 已生成单词 '{clean_token}' 的高清热力图 -> {save_path}")

    # ==========================================
    # 阶段 B：生成整体关注度 (Overall Focus) 高清热力图
    # ==========================================
    if valid_attn_list:
        overall_attn_raw = np.mean(valid_attn_list, axis=0)
        overall_attn = overall_attn_raw.reshape(7, 7)

        overall_attn = overall_attn - np.min(overall_attn)
        if np.max(overall_attn) > 0:
            overall_attn = overall_attn / np.max(overall_attn)

        # 🌟 关键修改：将整体热力图也放大到真实尺寸
        overall_heatmap = cv2.resize(overall_attn, (real_w, real_h))
        overall_heatmap = np.uint8(255 * overall_heatmap)
        overall_heatmap_color = cv2.applyColorMap(overall_heatmap, cv2.COLORMAP_JET)

        # 叠加到高清原图上
        overall_superimposed = overall_heatmap_color * 0.4 + orig_img * 0.6

        overall_save_path = os.path.join(args.output_dir, "attn_overall_focus.jpg")
        cv2.imwrite(overall_save_path, overall_superimposed)
        print(f"\n   🌟 【核心输出】已生成模型整体视觉焦点高清热力图 -> {overall_save_path}")

    print("\n🎉 全部生成完毕！快去 heatmaps 文件夹查看高清大图吧！")

if __name__ == "__main__":
    main()