# train_modular.py

import os
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# 导入我们需要的所有模块
from data import CrisisDataset, TextProcessor, get_transforms, DEFAULT_DATA_ROOT, TASK_CONFIG
from modules import (
    DenseNetVisualEncoder,
    ElectraTextEncoder,
    CrisisKANFusion,
    CrisisKANClassifier,
    ModularCrisisModel
)


# ==========================================
# 1. 参数配置中心 (The Cockpit)
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Modular CrisisKAN Model")

    # --- 基础配置 ---
    parser.add_argument('--task_name', type=str, default='task2', choices=['task1', 'task2'], help='任务名称')
    parser.add_argument('--run_name', type=str, default='experiment_01', help='本次实验的名称(用于保存文件)')
    parser.add_argument('--output_dir', type=str, default='./output_modular', help='模型保存路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子(复现性)')

    # --- 数据配置 ---
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT, help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=8, help='训练批次大小 (显存不够就调小)')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')

    # --- 训练超参数 ---
    parser.add_argument('--epochs', type=int, default=20, help='总训练轮数')
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='权重衰减')
    parser.add_argument('--patience', type=int, default=5, help='Early Stopping 耐心轮数')

    # --- 模型结构参数 (可自由定制) ---
    parser.add_argument('--visual_weights', type=str, default='../local_models/densenet201-c1103571.pth',
                        help='视觉模型权重路径')
    parser.add_argument('--text_model_path', type=str, default='../local_models/google/electra-base-discriminator',
                        help='文本模型路径')
    parser.add_argument('--proj_dim', type=int, default=100, help='融合投影维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout比率')

    return parser.parse_args()


# ==========================================
# 2. 工具函数
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ==========================================
# 3. 核心训练与验证循环
# ==========================================
def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(loader, desc=f"Train Epoch {epoch}")
    for batch in loop:
        # 1. 数据搬运到 GPU
        images = batch['image'].to(device)
        text_inputs = {k: v.to(device) for k, v in batch['text_tokens'].items()}
        labels = batch['label'].to(device)

        # 2. 前向传播
        optimizer.zero_grad()
        # 兼容 Dataset 字典格式，重新打包
        inputs = {'image': images, 'text_tokens': text_inputs}
        logits = model(inputs)

        # 3. 计算 Loss
        loss = criterion(logits, labels)

        # 4. 反向传播
        loss.backward()
        optimizer.step()

        # 5. 统计
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=correct / total)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            text_inputs = {k: v.to(device) for k, v in batch['text_tokens'].items()}
            labels = batch['label'].to(device)

            inputs = {'image': images, 'text_tokens': text_inputs}
            logits = model(inputs)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return total_loss / len(loader), acc, f1


# ==========================================
# 4. 主程序 (The Pilot)
# ==========================================
def main():
    args = parse_args()
    set_seed(args.seed)

    # 1. 准备目录和日志
    save_dir = os.path.join(args.output_dir, args.run_name)
    logger = setup_logger(save_dir)
    logger.info(f"🚀 Starting Experiment: {args.run_name}")
    logger.info(f"📝 Config: {args}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"⚙️ Device: {device}")

    # 2. 准备数据
    logger.info("📚 Loading Datasets...")
    text_proc = TextProcessor(model_name=args.text_model_path)
    train_transform = get_transforms(mode='train')
    eval_transform = get_transforms(mode='eval')

    train_set = CrisisDataset(args.data_root, args.task_name, 'train', train_transform, text_proc)
    dev_set = CrisisDataset(args.data_root, args.task_name, 'dev', eval_transform, text_proc)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = TASK_CONFIG[args.task_name]['num_classes']
    logger.info(f"📊 Task: {args.task_name} | Classes: {num_classes} | Train Size: {len(train_set)}")

    # 3. 组装模型 (Modular Assembly)
    logger.info("🏗️ Building Modular Model...")
    # A. 视觉编码器
    vis_enc = DenseNetVisualEncoder(weights_path=args.visual_weights)
    # B. 文本编码器
    txt_enc = ElectraTextEncoder(model_path=args.text_model_path)
    # C. 融合层
    fusion = CrisisKANFusion(
        visual_dim=vis_enc.output_dim,
        text_dim=txt_enc.output_dim,
        proj_dim=args.proj_dim
    )
    # D. 分类头
    cls_head = CrisisKANClassifier(
        input_dim=fusion.output_dim,
        num_classes=num_classes,
        dropout_rate=args.dropout
    )
    # E. 总装
    model = ModularCrisisModel(vis_enc, txt_enc, fusion, cls_head)
    model.to(device)

    # 4. 优化器与损失
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    # 5. 开始训练
    logger.info("🔥 Start Training...")
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        logger.info(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        # --- Validation ---
        val_loss, val_acc, val_f1 = evaluate(model, dev_loader, criterion, device)
        logger.info(f"[Epoch {epoch}] Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        scheduler.step(val_f1)

        # --- Save Best ---
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            save_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(model.state_dict(), save_path)
            logger.info(f"🌟 New Best Model Saved! (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(f"⏳ No improvement. Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            logger.info("🛑 Early Stopping triggered.")
            break

    logger.info("✅ Training Finished.")


if __name__ == "__main__":
    main()