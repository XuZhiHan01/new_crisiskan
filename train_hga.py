# train_hga.py

import os
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# 0. 新增：Focal Loss 实现
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (Tensor, optional): 类别权重，用于解决不平衡. Shape: (num_classes,)
            gamma (float): 聚焦参数，越大越关注难分类样本. Default: 2.0
            reduction (str): 'mean' | 'sum' | 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (Batch, Num_Classes) -> Logits
        # targets: (Batch) -> Labels

        # 1. 计算 Cross Entropy Loss (不归约，保留每个样本的 Loss)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 2. 计算 pt (概率)
        pt = torch.exp(-ce_loss)

        # 3. 计算 Focal Term: (1 - pt)^gamma
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 4. 应用 Alpha (类别权重)
        if self.alpha is not None:
            # 获取每个样本对应的 alpha
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # 5. 归约
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==========================================
# 1. 参数配置
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train HGA-Net with ResNet & BERTweet")

    # --- 基础配置 ---
    parser.add_argument('--task_name', type=str, default='task2', choices=['task1', 'task2', 'task3'])
    parser.add_argument('--run_name', type=str, default='hga_focal_loss_exp', help='实验名称')  # 改个名，方便区分
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
    logger.info(f"🚀 Starting HGA-Net (ResNet+BERTweet+FocalLoss): {args.run_name}")

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

    # ========================================================
    # ✨ 核心修改：使用 Focal Loss 并设置 Class Weights
    # ========================================================
    # 这里的权重是根据你的数据分布估算的：样本越少，权重越大
    # 0: infra (319) -> 1.0
    # 1: not_hum (849) -> 0.5 (最多)
    # 2: other (578) -> 0.8
    # 3: rescue (340) -> 1.0
    # 4: vehicle (19) -> 5.0
    # 5: affected (86) -> 4.0
    # 6: injured (41) -> 8.0
    # 7: missing (5) -> 15.0 (极少)

    class_weights = torch.tensor([
        1.0, 0.5, 0.8, 1.0, 5.0, 4.0, 8.0, 15.0
    ]).to(device)

    # 如果是 Task 1 或 Task 3，你需要根据它们的类别数调整 weights
    # 简单的做法是：如果不匹配 task2，就不传 alpha (退化为普通 Focal Loss)
    if num_classes != 8:
        print(
            f"⚠️ Warning: Class weights defined for 8 classes, but current task has {num_classes}. Disabling weighted alpha.")
        class_weights = None

    logger.info(f"⚖️ Using Focal Loss (gamma=2.0) with weights: {class_weights}")
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    # ... (检查并加载 best_model 的逻辑保持不变)
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

        # Save Best
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