"""
数据集加载模块
包含红外气体数据集的加载和预处理
"""
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def load_ids(train_file, test_file):
    """
    Args:
        train_file: 训练集 ID 文件路径
        test_file: 测试集 ID 文件路径
    
    Returns:
        train_ids: 训练集 ID 列表
        test_ids: 测试集 ID 列表
    """
    with open(train_file, 'r') as f:
        train_ids = [line.strip() for line in f.readlines()]
    with open(test_file, 'r') as f:
        test_ids = [line.strip() for line in f.readlines()]

    return train_ids, test_ids


class GasDataset(Dataset):
    """红外气体分割数据集
    
    数据集结构：
    - root/T/{id}.png: 红外图像（灰度图）
    - root/visual_labels/{id}.png: 分割标签（二值图）
    """
    def __init__(self, root, ids, transform=None):
        """
        Args:
            root: 数据集根目录
            ids: 样本 ID 列表
            transform: 图像变换（应用于输入图像）
        """
        self.root = root
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # 加载红外图像（灰度图）
        img = Image.open(f"{self.root}/T/{img_id}.png").convert("L")
        
        # 加载标签（二值图：白色为气体，黑色为背景）
        label = Image.open(f"{self.root}/visual_labels/{img_id}.png").convert("L")
        label = np.array(label)
        label = (label > 0).astype(np.float32)  # 白色为1，黑色为0
        label = torch.from_numpy(label).unsqueeze(0)  # [1, H, W]

        # 应用变换
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label
