# data/text_proc.py

import re
import torch
from transformers import ElectraTokenizer
from transformers import AutoTokenizer # 使用 AutoTokenizer 自动适配

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
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"#[^\s]+", "", text)
    text = re.sub(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()


class TextProcessor:
    def __init__(self, model_name='vinai/bertweet-base', max_len=128):
        # BERTweet 推荐 max_len=128
        print(f"Loading Tokenizer: {model_name}")

        # normalized=True 是 BERTweet 特有的需求，但用 AutoTokenizer + use_fast=False 通常比较稳
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, normalization=True)
        except:
            # 如果参数不支持，回退到默认
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.max_len = max_len

    def __call__(self, text):
        cleaned = clean_text(text)
        encoding = self.tokenizer(
            cleaned,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}