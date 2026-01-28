# train_hga.py

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

# 导入通用组件
from data import CrisisDataset, TextProcessor, get_transforms, DEFAULT_DATA_ROOT, TASK_CONFIG

# 1. 导入新模块 (ResNet & BERTweet)
from modules import (
    ResNetVisualEncoder,
    BERTweetTextEncoder,
    HGAFusion,
    CrisisKANClassifier,
    ModularCrisisModel
)


# ==========================================
# 1. 参数配置
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train HGA-Net with ResNet & BERTweet")

    # --- 基础配置 ---
    parser.add_argument('--task_name', type=str, default='task2', choices=['task1', 'task2', 'task3'])
    parser.add_argument('--run_name', type=str, default='hga_resnet_bertweet_exp', help='实验名称')
    parser.add_argument('--output_dir', type=str, default='./output_hga', help='保存路径')
    parser.add_argument('--seed', type=int, default=42)

    # --- 数据配置 ---
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)

    # --- 训练超参数 ---
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=5)

    # --- 模型路径配置 ---
    parser.add_argument('--visual_weights', type=str, default='../local_models/resnet50-0676ba61.pth',
                        help='本地 ResNet50 权重路径')

    # ✅ 修复了之前的拼写错误 (bertweet-basee -> bertweet-base)
    parser.add_argument('--text_model_path', type=str, default='../local_models/vinai/bertweet-base',
                        help='本地 BERTweet 文件夹路径')

    # --- HGA 结构参数 ---
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)

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
# 3. 训练循环
# ==========================================
def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(loader, desc=f"Train Epoch {epoch}")
    for batch in loop:
        images = batch['image'].to(device)
        text_inputs = {k: v.to(device) for k, v in batch['text_tokens'].items()}
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        inputs = {'image': images, 'text_tokens': text_inputs}
        logits = model(inputs)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

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

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), acc, f1


# ==========================================
# 4. 主程序
# ==========================================
def main():
    args = parse_args()
    set_seed(args.seed)

    save_dir = os.path.join(args.output_dir, args.run_name)
    logger = setup_logger(save_dir)
    logger.info(f"🚀 Starting HGA-Net (ResNet+BERTweet): {args.run_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"⚙️ Device: {device}")

    # 1. 数据准备
    logger.info("📚 Loading Datasets...")
    text_proc = TextProcessor(model_name=args.text_model_path)
    train_transform = get_transforms(mode='train')
    eval_transform = get_transforms(mode='eval')

    train_set = CrisisDataset(args.data_root, args.task_name, 'train', train_transform, text_proc)
    dev_set = CrisisDataset(args.data_root, args.task_name, 'dev', eval_transform, text_proc)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = TASK_CONFIG[args.task_name]['num_classes']

    # 2. 组装模型
    logger.info("🏗️ Assembling HGA-Net Components...")

    vis_weight = args.visual_weights if os.path.exists(args.visual_weights) else None
    vis_enc = ResNetVisualEncoder(weights_path=vis_weight, pretrained=True)

    txt_enc = BERTweetTextEncoder(model_path=args.text_model_path)

    fusion = HGAFusion(
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

    # 3. 训练配置
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    # ========================================================
    # ✨ 核心修改：检查并加载 best_model.pt
    # ========================================================
    best_f1 = 0.0
    patience_counter = 0
    best_model_path = os.path.join(save_dir, 'best_model.pt')

    if os.path.exists(best_model_path):
        logger.info("\n" + "=" * 40)
        logger.info(f"🔄 Found existing best model: {best_model_path}")
        logger.info("📥 Loading weights to resume training...")
        try:
            state_dict = torch.load(best_model_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info("✅ Weights loaded successfully!")

            # 关键步骤：先跑一次验证，确立基准线
            logger.info("📊 Evaluating baseline performance (please wait)...")
            _, _, init_f1 = evaluate(model, dev_loader, criterion, device)
            best_f1 = init_f1
            logger.info(f"🏁 Resuming with Baseline F1: {best_f1:.4f}")
            logger.info("=" * 40 + "\n")

        except Exception as e:
            logger.error(f"❌ Error loading checkpoint: {e}")
            logger.info("⚠️ Starting from scratch instead.")
            logger.info("=" * 40 + "\n")
    else:
        logger.info("🆕 No existing checkpoint found. Starting fresh training.")

    # 4. 开始训练
    logger.info("🔥 Start Training Loop...")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        logger.info(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        # Validation
        val_loss, val_acc, val_f1 = evaluate(model, dev_loader, criterion, device)
        logger.info(f"[Epoch {epoch}] Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        scheduler.step(val_f1)

        # Save Best (如果比基准线还好，就覆盖)
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
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