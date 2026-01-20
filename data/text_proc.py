# data/text_proc.py

import re
import torch
from transformers import ElectraTokenizer


def clean_text(text):
    """
    清洗推文文本: 去除 URL, 用户名, 非ASCII字符等
    (逻辑源自原 preprocess.py)
    """
    text = re.sub(r"http\S+", "", text)  # 去除链接
    text = re.sub(r"@[^\s]+", "", text)  # 去除 @用户
    text = re.sub(r"#[^\s]+", "", text)  # 去除 #话题
    text = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ", text)  # 去除特殊符号
    text = re.sub(r"\s{2,}", " ", text)  # 去除多余空格
    return text.strip().lower()


class TextProcessor:
    """
    文本处理器：负责清洗和分词
    """

    def __init__(self, model_name='../local_models/google/electra-base-discriminator', max_len=512):
        try:
            self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        except:
            print(f"Warning: Local tokenizer not found at {model_name}, trying huggingface...")
            self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')

        self.max_len = max_len

    def __call__(self, text):
        """
        输入: 原始文本字符串
        输出: Token字典 {input_ids, attention_mask, ...}
        """
        cleaned = clean_text(text)
        encoding = self.tokenizer(
            cleaned,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        # 降维: (1, seq_len) -> (seq_len)
        return {k: v.squeeze(0) for k, v in encoding.items()}