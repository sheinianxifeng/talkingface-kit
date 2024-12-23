import os
import random


def generate_data_files(data_root, output_dir, train_split=0.8):
    """
    生成训练集和测试集的txt文件。
    :param data_root: 数据集根目录 (如 ./data/HDTF)
    :param output_dir: 输出目录 (如 ./data)
    :param train_split: 训练集比例，默认为 80%
    """
    # 获取所有图片子目录名称
    image_dir = os.path.join(data_root, 'images')
    subdirs = [d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))]

    # 打乱顺序
    random.shuffle(subdirs)

    # 按比例划分为训练集和测试集
    split_idx = int(len(subdirs) * train_split)
    train_dirs = subdirs[:split_idx]
    test_dirs = subdirs[split_idx:]

    # 保存到文件
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'data_train.txt'), 'w') as f:
        f.writelines(f"{d}\n" for d in train_dirs)
    with open(os.path.join(output_dir, 'data_test.txt'), 'w') as f:
        f.writelines(f"{d}\n" for d in test_dirs)

    print(f"训练集和测试集已生成：\n- 训练集：{len(train_dirs)} 条\n- 测试集：{len(test_dirs)} 条")


# 示例调用
data_root = 'D:/DiffTalk/data/HDTF'  # 数据集根目录
output_dir = 'D:/DiffTalk/data'      # 输出目录
generate_data_files(data_root, output_dir)
