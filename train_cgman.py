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
from transformers import get_cosine_schedule_with_warmup
# 导入通用组件
from data import CrisisDataset, TextProcessor, get_transforms, DEFAULT_DATA_ROOT, TASK_CONFIG

# 导入新模块 (注意这里引入了 CGMANFusion)
from modules import (
    ResNetVisualEncoder,
    BERTweetTextEncoder,
    ConvNextVisualEncoder,
    DebertaTextEncoder,
    CGMANFusion,  # <--- 引入新的 C-GMAN 融合模块
    CrisisKANClassifier,
    ModularCrisisModel
)



# ==========================================
# 1. 参数配置
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train C-GMAN Model")
    parser.add_argument('--task_name', type=str, default='task3', choices=['task1', 'task2', 'task3'])
    parser.add_argument('--run_name', type=str, default='task3_lamda_0.2', help='实验名称')
    parser.add_argument('--output_dir', type=str, default='./output_cgman', help='保存路径')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--patience', type=int, default=2)
    # 🌟 新增：梯度累加步数
    parser.add_argument('--accumulation_steps', type=int, default=8,
                        help='梯度累加步数 (实际Batch = batch_size * steps)')

    # 🌟 新增：对比损失的权重系数 (论文可做消融实验)
    parser.add_argument('--lambda_cl', type=float, default=0.15, help='对比损失的权重')

    parser.add_argument('--visual_weights', type=str, default='../local_models/resnet50-0676ba61.pth')
    # 指向你刚刚下载的 DeBERTa 本地文件夹
    parser.add_argument('--text_model_path', type=str, default='../local_models/deberta-v3-base')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.4)

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
# 🌟 修改：接收 accumulation_steps 参数
def train_epoch(model, loader, optimizer, criterion_ce, device, epoch,
                lambda_cl, accumulation_steps, scheduler, scaler):
    model.train()
    total_loss, total_ce, total_cl = 0, 0, 0
    correct, total = 0, 0

    # 🌟 关键 1：在 epoch 开始前清空梯度
    optimizer.zero_grad()

    loop = tqdm(loader, desc=f"Train Epoch {epoch}")
    # 🌟 修改：使用 enumerate 获取当前 batch 的索引 i
    for i, batch in enumerate(loop):
        images = batch['image'].to(device)
        text_inputs = {k: v.to(device) for k, v in batch['text_tokens'].items()}
        labels = batch['label'].to(device)

        inputs = {'image': images, 'text_tokens': text_inputs}

        # 🌟 关键修改 1：开启 autocast 混合精度上下文
        with torch.cuda.amp.autocast():
            # 1. 接收主分类预测(logits)和两个辅助分类预测(aux_v_logits, aux_t_logits)
            logits, aux_v_logits, aux_t_logits = model(inputs, return_features=True)

            # 2. 计算主分支的交叉熵损失 (这是融合后最终的预测)
            loss_ce = criterion_ce(logits, labels)

            # 3. 计算视觉和文本单模态的独立交叉熵损失 (防止模态偷懒)
            loss_aux_v = criterion_ce(aux_v_logits, labels)
            loss_aux_t = criterion_ce(aux_t_logits, labels)

            # 4. 把两个辅助损失加起来。
            # (💡 技巧：这里我故意继续用 loss_cl 这个变量名，这样你下面打印 tqdm 进度条的代码 total_cl += loss_cl.item() 就一行都不用改了！)
            loss_cl = loss_aux_v + loss_aux_t

            # 5. 联合损失加权：主损失 + λ * (视觉辅助 + 文本辅助)
            loss = loss_ce + lambda_cl * loss_cl
            loss = loss / accumulation_steps

        # 🌟 关键修改 2：使用 scaler 放大 loss 并反向传播
        scaler.scale(loss).backward()

        # 🌟 关键 3：只有当达到累加步数，或者到了最后一个 batch 时，才真正更新权重并清空梯度
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            # 🌟 关键修改 3：在裁剪梯度前，必须先 unscale
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 🌟 关键修改 4：使用 scaler 步进优化器
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # 统计指标 (注意把 loss 乘回来，以免打印出的数值偏小造成误解)
        total_loss += loss.item() * accumulation_steps
        total_ce += loss_ce.item()
        total_cl += loss_cl.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # 进度条显示当前的真实 loss
        loop.set_postfix(
            loss=loss.item() * accumulation_steps,
            ce=loss_ce.item(),
            cl=loss_cl.item(),
            acc=correct / total
        )

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

            # 1. 同样地，解包出三个预测值
            logits, aux_v_logits, aux_t_logits = model(inputs, return_features=True)

            # 2. 计算三个独立的交叉熵损失
            loss_ce = criterion_ce(logits, labels)
            loss_aux_v = criterion_ce(aux_v_logits, labels)
            loss_aux_t = criterion_ce(aux_t_logits, labels)

            # 3. 组合联合损失
            loss_cl = loss_aux_v + loss_aux_t
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

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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
    vis_enc = ConvNextVisualEncoder()
    txt_enc = DebertaTextEncoder(model_path=args.text_model_path)
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

    model = ModularCrisisModel(vis_enc, txt_enc, fusion, cls_head,
                               num_classes=num_classes, embed_dim=args.embed_dim)
    model.to(device)

    # 优化器
    # 区分预训练模块和随机初始化模块
    # 🌟 修改 2：把辅助分类头的参数加入到 fresh_params 中，让它们能被更新
    pretrained_params = list(model.visual_encoder.parameters()) + list(model.text_encoder.parameters())
    fresh_params = list(model.fusion_module.parameters()) + list(model.classifier.parameters())
    if hasattr(model, 'aux_vis_head') and model.aux_vis_head is not None:
        fresh_params += list(model.aux_vis_head.parameters()) + list(model.aux_txt_head.parameters())

    optimizer = optim.AdamW([
        {'params': pretrained_params, 'lr': 5e-6},
        {'params': fresh_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    # 计算总步数 (Total steps)
    total_steps = len(train_loader) // args.accumulation_steps * args.epochs
    # 预热步数设为总步数的 10%
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 🌟 新增：混合精度梯度缩放器
    scaler = torch.cuda.amp.GradScaler()

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
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion_ce, device, epoch,
            args.lambda_cl, args.accumulation_steps, scheduler, scaler
        )
        logger.info(f"[Epoch {epoch}] Train Total Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        val_loss, val_acc, val_f1 = evaluate(model, dev_loader, criterion_ce, device, args.lambda_cl)
        logger.info(f"[Epoch {epoch}] Val Total Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")



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