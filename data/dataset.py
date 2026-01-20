# data/dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from .config import TASK_CONFIG, DEFAULT_DATA_ROOT


class CrisisDataset(Dataset):
    def __init__(self,
                 root_dir=DEFAULT_DATA_ROOT,
                 task_name='task2',
                 phase='train',
                 transform=None,
                 text_processor=None):
        """
        Args:
            root_dir: 数据集根目录
            task_name: 'task1' 或 'task2'
            phase: 'train', 'dev', 'test'
            transform: 图像预处理函数
            text_processor: 文本处理实例
        """
        self.root_dir = root_dir
        self.transform = transform
        self.text_processor = text_processor

        # 1. 获取任务配置
        if task_name not in TASK_CONFIG:
            raise ValueError(f"Unknown task: {task_name}")

        self.task_info = TASK_CONFIG[task_name]
        self.label_map = self.task_info['label_map']

        # 2. 构建 TSV 文件路径
        # 格式示例: task_humanitarian_text_img_train.tsv
        tsv_name = f"task_{self.task_info['name']}_text_img_{phase}.tsv"
        self.tsv_path = os.path.join(root_dir, 'crisismmd_datasplit_all', tsv_name)

        # 3. 读取数据
        self.data_list = self._read_tsv(self.tsv_path)
        print(f"[{phase.upper()}] Loaded {len(self.data_list)} samples from {tsv_name}")

    def _read_tsv(self, path):
        data = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"TSV file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # 跳过表头
            for line in lines:
                line = line.strip()
                if not line: continue

                # 解析 TSV 行 (原代码逻辑)
                parts = line.split('\t')
                # 假设格式: event, tweet_id, image_id, text, image_path, label, ... final_text
                # 我们只取需要的关键字段
                image_rel_path = parts[4]
                label_str = parts[5]
                final_text = parts[-1]  # 最后一列通常是处理过的 clean text 或 原始 text

                # 过滤掉不在 label_map 里的脏数据
                if label_str in self.label_map:
                    data.append({
                        'image_path': os.path.join(self.root_dir, image_rel_path),
                        'text': final_text,
                        'label': self.label_map[label_str]
                    })
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        # A. 处理图片
        image = Image.open(item['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # B. 处理文本
        text_data = item['text']
        if self.text_processor:
            text_data = self.text_processor(item['text'])

        # C. 处理标签
        label = torch.tensor(item['label'], dtype=torch.long)

        return {
            'image': image,
            'text_tokens': text_data,  # 这是一个字典 {input_ids, ...}
            'label': label,
            'raw_text': item['text'],  # 保留原始文本方便调试
            'image_path': item['image_path']
        }