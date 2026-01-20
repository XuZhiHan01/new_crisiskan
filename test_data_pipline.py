# test_data_loading.py

from torch.utils.data import DataLoader
from data.dataset import CrisisDataset
from data.transforms import get_transforms
from data.text_proc import TextProcessor


def main():
    # 1. 准备组件
    print("Initializing Text Processor...")
    text_proc = TextProcessor()

    print("Initializing Transforms...")
    train_transform = get_transforms(mode='train')

    # 2. 实例化 Dataset
    print("Loading Dataset...")
    # 注意：请确保你的 datasets 文件夹路径正确
    dataset = CrisisDataset(
        root_dir='../datasets/CrisisMMD_v2.0',
        task_name='task2',
        phase='train',
        transform=train_transform,
        text_processor=text_proc
    )

    # 3. 实例化 DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 4. 试运行一个 Batch
    print("\nTesting one batch...")
    for batch in dataloader:
        imgs = batch['image']
        tokens = batch['text_tokens']
        labels = batch['label']

        print(f"Image Batch Shape: {imgs.shape}")  # 应该 (4, 3, 224, 224)
        print(f"Text Input IDs Shape: {tokens['input_ids'].shape}")  # 应该 (4, 512)
        print(f"Labels: {labels}")

        break

    print("\n✅ Data loading pipeline is ready!")


if __name__ == "__main__":
    main()