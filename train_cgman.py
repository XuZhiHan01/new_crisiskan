# train_cgman.py

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

# 导入新模块 (注意这里引入了 CGMANFusion)
from modules import (
    ResNetVisualEncoder,
    BERTweetTextEncoder,
    CGMANFusion,  # <--- 引入新的 C-GMAN 融合模块
    CrisisKANClassifier,
    ModularCrisisModel
)


# ==========================================
# 🌟 新增：跨模态对比损失函数 (InfoNCE) 🌟
# ==========================================
def contrastive_loss(v_feat, t_feat, temperature=0.07):
    """
    计算图像全局特征和文本全局特征的对比损失。
    这在论文中可以大书特书：Semantic-Anchored Contrastive Pre-alignment
    """
    # 1. 特征 L2 归一化
    v_feat = F.normalize(v_feat, dim=-1)
    t_feat = F.normalize(t_feat, dim=-1)

    # 2. 计算余弦相似度矩阵 (B, B)
    logits = torch.matmul(v_feat, t_feat.T) / temperature

    # 3. 构建目标标签 (对角线上的元素互为正样本)
    labels = torch.arange(logits.size(0), device=logits.device)

    # 4. 对称交叉熵 (Image2Text 和 Text2Image)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2.0


# ==========================================
# 1. 参数配置
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train C-GMAN Model")
    parser.add_argument('--task_name', type=str, default='task1', choices=['task1', 'task2', 'task3'])
    parser.add_argument('--run_name', type=str, default='cgman_resnet_bertweet_exp', help='实验名称')
    parser.add_argument('--output_dir', type=str, default='./output_cgman', help='保存路径')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=5)

    # 🌟 新增：对比损失的权重系数 (论文可做消融实验)
    parser.add_argument('--lambda_cl', type=float, default=0.1, help='对比损失的权重')

    parser.add_argument('--visual_weights', type=str, default='../local_models/resnet50-0676ba61.pth')
    parser.add_argument('--text_model_path', type=str, default='../local_models/vinai/bertweet-base')

    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)

    return parser.parse_args()


# 工具函数
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
        level=logging.INFO,
        handlers=[logging.FileHandler(os.path.join(output_dir, 'train.log')), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


# ==========================================
# 3. 训练循环 (联合优化)
# ==========================================
def train_epoch(model, loader, optimizer, criterion_ce, device, epoch, lambda_cl):
    model.train()
    total_loss, total_ce, total_cl = 0, 0, 0
    correct, total = 0, 0

    loop = tqdm(loader, desc=f"Train Epoch {epoch}")
    for batch in loop:
        images = batch['image'].to(device)
        text_inputs = {k: v.to(device) for k, v in batch['text_tokens'].items()}
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        inputs = {'image': images, 'text_tokens': text_inputs}

        # 🌟 关键修改：开启 return_features=True 提取全局特征
        logits, v_global, t_global = model(inputs, return_features=True)

        # 1. 分类损失 (Cross Entropy)
        loss_ce = criterion_ce(logits, labels)

        # 2. 对比损失 (Contrastive Loss)
        loss_cl = contrastive_loss(v_global, t_global)

        # 3. 联合损失 (Joint Loss)
        loss = loss_ce + lambda_cl * loss_cl

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_cl += loss_cl.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), ce=loss_ce.item(), cl=loss_cl.item(), acc=correct / total)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion_ce, device, lambda_cl):
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
            logits, v_global, t_global = model(inputs, return_features=True)

            loss_ce = criterion_ce(logits, labels)
            loss_cl = contrastive_loss(v_global, t_global)
            loss = loss_ce + lambda_cl * loss_cl

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

    # 动态将 task_name 插入到路径中，例如: ./output_cgman/task2/cgman_resnet_bertweet_exp
    save_dir = os.path.join(args.output_dir, args.task_name, args.run_name)
    logger = setup_logger(save_dir)
    logger.info(f"🚀 Starting C-GMAN (Contrastive-Guided Gated Network): {args.run_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"⚙️ Device: {device}")

    # 数据准备
    logger.info("📚 Loading Datasets...")
    text_proc = TextProcessor(model_name=args.text_model_path)
    train_transform = get_transforms(mode='train')
    eval_transform = get_transforms(mode='eval')

    train_set = CrisisDataset(args.data_root, args.task_name, 'train', train_transform, text_proc)
    dev_set = CrisisDataset(args.data_root, args.task_name, 'dev', eval_transform, text_proc)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = TASK_CONFIG[args.task_name]['num_classes']

    # 组装 C-GMAN 模型
    logger.info("🏗️ Assembling C-GMAN Components...")
    vis_weight = args.visual_weights if os.path.exists(args.visual_weights) else None
    vis_enc = ResNetVisualEncoder(weights_path=vis_weight, pretrained=True)
    txt_enc = BERTweetTextEncoder(model_path=args.text_model_path)

    # 🌟 关键修改：使用 CGMANFusion
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

    # 优化器
    # 区分预训练模块和随机初始化模块
    pretrained_params = list(model.visual_encoder.parameters()) + list(model.text_encoder.parameters())
    fresh_params = list(model.fusion_module.parameters()) + list(model.classifier.parameters())

    optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': args.lr},  # 例如: 1e-5 (保持微调步调)
        {'params': fresh_params, 'lr': args.lr * 10.0}  # 例如: 1e-4 (加速融合层收敛)
    ], weight_decay=args.weight_decay)
    criterion_ce = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

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

            # 关键步骤：先跑一次验证，确立当前的基准线，防止刚加载就被覆盖
            logger.info("📊 Evaluating baseline performance (please wait)...")
            _, _, init_f1 = evaluate(model, dev_loader, criterion_ce, device, args.lambda_cl)
            best_f1 = init_f1
            logger.info(f"🏁 Resuming with Baseline F1: {best_f1:.4f}")
            logger.info("=" * 40 + "\n")

        except Exception as e:
            logger.error(f"❌ Error loading checkpoint: {e}")
            logger.info("⚠️ Starting from scratch instead.")
            logger.info("=" * 40 + "\n")
    else:
        logger.info("🆕 No existing checkpoint found. Starting fresh training.")

    logger.info("🔥 Start Joint Training Loop...")

    for epoch in range(1, args.epochs + 1):
        # 传入 lambda_cl 控制对比损失占比
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion_ce, device, epoch, args.lambda_cl)
        logger.info(f"[Epoch {epoch}] Train Total Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        val_loss, val_acc, val_f1 = evaluate(model, dev_loader, criterion_ce, device, args.lambda_cl)
        logger.info(f"[Epoch {epoch}] Val Total Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"🌟 New Best C-GMAN Model Saved! (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(f"⏳ No improvement. Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            logger.info("🛑 Early Stopping triggered.")
            break

    logger.info("✅ Training Finished.")


if __name__ == "__main__":
    main()