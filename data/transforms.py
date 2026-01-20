# data/transforms.py

from PIL import Image
import torchvision.transforms as transforms


def expand2square(pil_img, background_color):
    """
    将图片填充为正方形 (保持长宽比)
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def get_transforms(mode='train', image_size=224):
    """
    获取图像预处理流水线
    mode: 'train' (包含随机增强) 或 'eval' (仅缩放归一化)
    """
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # 基础步骤：填充正方形 -> 缩放
    base_transforms = [
        transforms.Lambda(lambda img: expand2square(img, (0, 0, 0))),
        transforms.Resize((image_size, image_size)),
    ]

    if mode == 'train':
        # 训练集：加入随机裁剪、翻转等数据增强
        base_transforms.extend([
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomCrop((image_size, image_size)),  # 或者用 RandomResizedCrop
            transforms.ToTensor(),
            normalize
        ])
    else:
        # 测试集：直接转 Tensor
        base_transforms.extend([
            transforms.ToTensor(),
            normalize
        ])

    return transforms.Compose(base_transforms)