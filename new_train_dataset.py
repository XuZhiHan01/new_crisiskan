import os
import pandas as pd


def process_datasets(base_path, add_ratio, random_seed=42):
    """
    向训练集中添加一定百分比的测试集，并生成新的训练集和测试集。

    :param base_path: 数据集存放的基础路径
    :param add_ratio: 从测试集中抽取的比例 (例如 0.2 表示 20%)
    :param random_seed: 随机种子，确保每次采样的结果一致
    """

    # 定义三个任务
    tasks = ['damage', 'humanitarian', 'informative']

    print(f"开始处理数据集，测试集抽取比例为: {add_ratio * 100}%")

    for task in tasks:
        # 构建原始文件路径
        train_file = os.path.join(base_path, f'task_{task}_text_img_train.tsv')
        test_file = os.path.join(base_path, f'task_{task}_text_img_test.tsv')

        # 构建新文件路径 (添加了 _new 后缀，避免直接覆盖原文件)
        new_train_file = os.path.join(base_path, f'task_{task}_text_img_train_new.tsv')
        new_test_file = os.path.join(base_path, f'task_{task}_text_img_test_new.tsv')

        # 读取 TSV 文件
        try:
            df_train = pd.read_csv(train_file, sep='\t')
            df_test = pd.read_csv(test_file, sep='\t')
        except FileNotFoundError as e:
            print(f"找不到文件: {e.filename}，请检查路径。")
            continue

        # 从测试集中随机采样指定比例的数据
        df_test_sampled = df_test.sample(frac=add_ratio, random_state=random_seed)

        # 将测试集剩余的部分作为新的测试集
        df_test_new = df_test.drop(df_test_sampled.index)

        # 将采样出的数据添加到原训练集中
        df_train_new = pd.concat([df_train, df_test_sampled], ignore_index=True)

        # 随机打乱新的训练集 (可选，但在深度学习训练中推荐)
        #df_train_new = df_train_new.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

        # 保存为新的 TSV 文件
        df_train_new.to_csv(new_train_file, sep='\t', index=False)
        df_test_new.to_csv(new_test_file, sep='\t', index=False)

        print(f"任务 [{task}]:")
        print(f"  - 原训练集大小: {len(df_train)}")
        print(f"  - 原测试集大小: {len(df_test)}")
        print(f"  - 转移数据量:   {len(df_test_sampled)}")
        print(f"  - 新训练集保存至 -> {new_train_file} (大小: {len(df_train_new)})")
        print(f"  - 新测试集保存至 -> {new_test_file} (大小: {len(df_test_new)})\n")


if __name__ == '__main__':
    # 设置路径
    dataset_path = '/home/tSdu/xzh/crisisKAN/crisiskan/datasets/settingA/'

    # 设置超参数：你想从测试集中抽出多少比例放到训练集里 (例如 0.15 代表 15%)
    RATIO_TO_MOVE = 0.58

    process_datasets(dataset_path, RATIO_TO_MOVE)