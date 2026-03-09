# test_cgman.py

import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, f1_score

# 导入通用组件
from data import CrisisDataset, TextProcessor, get_transforms, DEFAULT_DATA_ROOT, TASK_CONFIG

# ✅ 导入 C-GMAN 专属组件
from modules import (
    ResNetVisualEncoder,
    BERTweetTextEncoder,
    CGMANFusion,          # <--- 关键：使用 C-GMAN 融合
    CrisisKANClassifier,
    ModularCrisisModel,
    ConvNextVisualEncoder,
    DebertaTextEncoder
)

def parse_args():
    parser = argparse.ArgumentParser(description="Test C-GMAN (Contrastive-Guided Gated Network)")

    # --- 基础配置 ---
    parser.add_argument('--task_name', type=str, default='task1', choices=['task1', 'task2', 'task3'])
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)

    # --- 关键：模型路径 (请修改为你训练时 run_name 生成的目录) ---
    parser.add_argument('--checkpoint_path', type=str,
                        default='./output_cgman/task1/cgman_resnet_bertweet_exp02/best_model.pt',
                        help='训练好的 best_model.pt 路径')
    parser.add_argument('--output_dir', type=str, default='./test_results_cgman')

    # --- 文本模型路径 ---
    parser.add_argument('--text_model_path', type=str, default='../local_models/deberta-v3-base')

    # --- 模型结构参数 (必须与 train_cgman.py 保持完全一致！) ---
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)

    return parser.parse_args()


def run_test(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    print(f"Start Testing on {len(loader.dataset)} samples...")

    with torch.no_grad():
        loop = tqdm(loader, desc="Testing C-GMAN")
        for batch in loop:
            images = batch['image'].to(device)
            text_inputs = {k: v.to(device) for k, v in batch['text_tokens'].items()}
            labels = batch['label'].to(device)

            inputs = {'image': images, 'text_tokens': text_inputs}

            # 这里的推理，不需要 return_features=True，因为只做预测
            logits_normal = model(inputs)

            # 🌟 黑科技：2. 图像水平翻转后再推理一次
            flipped_images = torch.flip(images, dims=[3])  # 沿宽度维度翻转
            inputs_flipped = {'image': flipped_images, 'text_tokens': text_inputs}
            logits_flipped = model(inputs_flipped)

            # 🌟 3. 取两次预测的平均值作为最终决定
            logits = (logits_normal + logits_flipped) / 2.0

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ Device: {device}")

    # 1. 准备数据
    print("📚 Loading Test Dataset...")
    text_proc = TextProcessor(model_name=args.text_model_path)
    eval_transform = get_transforms(mode='eval')

    test_set = CrisisDataset(args.data_root, args.task_name, 'test', eval_transform, text_proc)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = TASK_CONFIG[args.task_name]['num_classes']
    label_map = TASK_CONFIG[args.task_name]['label_map']
    id_to_label = {v: k for k, v in label_map.items()}
    target_names = [id_to_label[i] for i in range(num_classes)]

    # 2. 组装 C-GMAN 模型
    print("🏗️ Re-building C-GMAN Architecture...")
    vis_enc = ConvNextVisualEncoder()
    txt_enc = DebertaTextEncoder(model_path=args.text_model_path)
    fusion = CGMANFusion(
        visual_dim=vis_enc.output_dim,
        text_dim=txt_enc.output_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        layers=args.layers
    )

    cls_head = CrisisKANClassifier(
        input_dim=fusion.output_dim,
        num_classes=num_classes,
        dropout_rate=args.dropout
    )

    model = ModularCrisisModel(vis_enc, txt_enc, fusion, cls_head)
    model.to(device)

    # 3. 加载权重
    print(f"📥 Loading Checkpoint from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path} \n请先确保 train_cgman.py 运行完毕并保存了模型！")

    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("✅ Weights Loaded Successfully!")

    # 4. 运行测试
    true_labels, pred_labels = run_test(model, test_loader, device)

    # ==========================================
    # 5. 生成报告并格式化输出
    # ==========================================
    acc = accuracy_score(true_labels, pred_labels)
    weighted_f1 = f1_score(true_labels, pred_labels, average='weighted')
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')

    # 将报告内容拼装成字符串，方便既打印到屏幕，又写入文件
    report_str = f"Accuracy:    {acc:.4f}\n"
    report_str += f"Weighted F1: {weighted_f1:.4f}\n"
    report_str += f"Macro F1:    {macro_f1:.4f}\n"
    report_str += "-" * 50 + "\n"
    report_str += classification_report(true_labels, pred_labels, labels=list(range(num_classes)),
                                        target_names=target_names, digits=4)
    report_str += "\n" + "=" * 50

    # 打印到控制台
    print("\n" + "=" * 50)
    print(f"🏆 C-GMAN TEST REPORT ({args.task_name.upper()})")
    print("=" * 50)
    print(report_str)

    # ==========================================
    # 6. 保存所有结果 (动态按 task_name 建立文件夹)
    # ==========================================
    final_output_dir = os.path.join(args.output_dir, args.task_name)
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)

    # 6.1 保存纯文本评估报告 (.txt)
    metrics_path = os.path.join(final_output_dir, 'metrics_report.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"🏆 C-GMAN TEST REPORT ({args.task_name.upper()})\n")
        f.write("=" * 50 + "\n")
        f.write(report_str + "\n")
    print(f"📝 Metrics report saved to: {metrics_path}")

    # 6.2 保存 prediction.csv (纯数字)
    pred_path = os.path.join(final_output_dir, 'prediction.csv')
    with open(pred_path, 'w') as f:
        for p in pred_labels:
            f.write(f"{p}\n")
    print(f"💾 Predictions saved to: {pred_path}")

    # 6.3 保存带标签的详细 csv (方便人工查看)
    df = pd.DataFrame({
        'True Label ID': true_labels,
        'Pred Label ID': pred_labels,
        'True Label Name': [id_to_label[i] for i in true_labels],
        'Pred Label Name': [id_to_label[i] for i in pred_labels]
    })
    detailed_path = os.path.join(final_output_dir, 'test_predictions_detailed.csv')
    df.to_csv(detailed_path, index=False)
    print(f"💾 Detailed results saved to: {detailed_path}")

if __name__ == "__main__":
    main()