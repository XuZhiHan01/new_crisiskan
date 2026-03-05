# test_modular.py

import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, f1_score

# 复用你已经写好的模块
from data import CrisisDataset, TextProcessor, get_transforms, DEFAULT_DATA_ROOT, TASK_CONFIG
from modules import (
    DenseNetVisualEncoder,
    ElectraTextEncoder,
    HGAFusion,
    CrisisKANClassifier,
    ModularCrisisModel
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test Modular CrisisKAN Model")

    # --- 基础配置 ---
    parser.add_argument('--task_name', type=str, default='task2', choices=['task1', 'task2', 'task3'], help='任务名称')
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT, help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=16, help='测试批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')

    # --- 关键：模型路径 ---
    # 这里默认填你训练脚本输出的那个 best_model.pt 的位置
    parser.add_argument('--checkpoint_path', type=str, required=True, help='训练好的 best_model.pt 路径')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='测试结果保存目录')

    # --- 模型结构参数 (必须与训练时保持一致！) ---
    # 如果你训练时改了 proj_dim，这里也要改，否则加载权重会报错
    parser.add_argument('--proj_dim', type=int, default=100, help='融合投影维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout比率')
    parser.add_argument('--text_model_path', type=str, default='../local_models/google/electra-base-discriminator',
                        help='文本模型路径(用于初始化tokenizer)')

    return parser.parse_args()


def run_test(model, loader, device, num_classes, label_map):
    model.eval()
    all_preds = []
    all_labels = []

    print(f"Start Testing on {len(loader.dataset)} samples...")

    with torch.no_grad():
        loop = tqdm(loader, desc="Testing")
        for batch in loop:
            # 1. 搬运数据
            images = batch['image'].to(device)
            text_inputs = {k: v.to(device) for k, v in batch['text_tokens'].items()}
            labels = batch['label'].to(device)

            inputs = {'image': images, 'text_tokens': text_inputs}

            # 2. 推理
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)

            # 3. 收集结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙Device: {device}")

    # 1. 准备数据 (Phase='test')
    print("Loading Test Dataset...")
    text_proc = TextProcessor(model_name=args.text_model_path)
    eval_transform = get_transforms(mode='eval')  # 注意这里用 eval 模式，不进行随机剪裁

    test_set = CrisisDataset(args.data_root, args.task_name, 'test', eval_transform, text_proc)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = TASK_CONFIG[args.task_name]['num_classes']
    label_map = TASK_CONFIG[args.task_name]['label_map']
    # 反转 label_map 以便打印类别名称: {0: 'damage', 1: '...'}
    id_to_label = {v: k for k, v in label_map.items()}
    target_names = [id_to_label[i] for i in range(num_classes)]

    # 2. 组装模型 (结构必须完全一致)
    print("🏗Re-building Model Architecture...")
    # 注意：这里不需要加载预训练权重路径，因为我们马上要加载 checkpoint
    vis_enc = DenseNetVisualEncoder(weights_path=None)
    txt_enc = ElectraTextEncoder(model_path=args.text_model_path)

    fusion = HGAFusion(
        visual_dim=vis_enc.output_dim,
        text_dim=txt_enc.output_dim
    )

    cls_head = CrisisKANClassifier(
        input_dim=fusion.output_dim,
        num_classes=num_classes,
        dropout_rate=args.dropout
    )

    model = ModularCrisisModel(vis_enc, txt_enc, fusion, cls_head)
    model.to(device)

    # 3. 加载训练好的权重
    print(f"Loading Checkpoint from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")

    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Weights Loaded Successfully!")

    # 4. 运行测试
    true_labels, pred_labels = run_test(model, test_loader, device, num_classes, label_map)

    # 5. 生成报告
    print("\n" + "=" * 50)
    print("TEST REPORT")
    print("=" * 50)

    acc = accuracy_score(true_labels, pred_labels)
    weighted_f1 = f1_score(true_labels, pred_labels, average='weighted')

    print(f"Accuracy:    {acc:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("-" * 50)
    print(classification_report(true_labels, pred_labels, target_names=target_names, digits=4))
    print("=" * 50)

    # 6. 保存预测结果 (可选，方便后续分析)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 保存 prediction.csv (纯数字，兼容之前的 calc_metric.py)
    pred_path = os.path.join(args.output_dir, 'prediction.csv')
    with open(pred_path, 'w') as f:
        for p in pred_labels:
            f.write(f"{p}\n")
    print(f"Predictions saved to: {pred_path}")

    # 保存带标签的详细 csv (方便人工查看)
    df = pd.DataFrame({
        'True Label ID': true_labels,
        'Pred Label ID': pred_labels,
        'True Label Name': [id_to_label[i] for i in true_labels],
        'Pred Label Name': [id_to_label[i] for i in pred_labels]
    })
    detailed_path = os.path.join(args.output_dir, 'test_predictions_detailed.csv')
    df.to_csv(detailed_path, index=False)
    print(f"Detailed results saved to: {detailed_path}")


if __name__ == "__main__":
    main()