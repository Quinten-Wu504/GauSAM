from prepare import pair_directories
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from .utils import apply_transform

class MyDataset(Dataset):
    def __init__(self, data_root, train=True, resize_size=(384, 384), label_resize_size=None):
        """
        Args:
            data_root (str): 数据集根目录
            train (bool): 是否为训练集
            resize_size (tuple): 图像缩放尺寸，可以通过设置较小尺寸实现降采样
            label_resize_size (tuple, optional): 标签缩放尺寸，默认与 `resize_size` 相同
        """
        self.train = train
        self.data_root = data_root
        self.resize_size = resize_size
        self.label_resize_size = label_resize_size if label_resize_size else resize_size

        split = "train" if train else "val"
        pairs = pair_directories(Path(data_root) / split / "images", Path(data_root) / split / "masks")
        if not pairs:
            raise ValueError(f"No complete image/mask pairs at {data_root}")
        self.img_files = [str(pair[0]) for pair in pairs]
        self.gt_files = [str(pair[1]) for pair in pairs]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = cv2.imread(self.img_files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.resize_size)
        gt2D = cv2.imread(self.gt_files[index], cv2.IMREAD_GRAYSCALE)
        gt2D = cv2.resize(gt2D, self.label_resize_size, interpolation=cv2.INTER_NEAREST)
        gt2D = ((gt2D > (0 if gt2D.max() <= 1 else 127)) * 255).astype(np.uint8)
        if self.train:
            original_img, original_mask = img, gt2D
            for _ in range(10):
                candidate_img, candidate_mask = apply_transform(original_img, original_mask)
                if not np.any(original_mask) or np.any(candidate_mask):
                    img, gt2D = candidate_img, candidate_mask
                    break
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        gt2D = gt2D.astype(np.uint8) / 255

        # 计算边界框（Bounding Box）
        y_indices, x_indices = np.where(gt2D > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            x_min, x_max, y_min, y_max = 0, 0, 0, 0
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
        # 加入随机扰动
        if self.train:
            H, W = gt2D.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        box = np.array([x_min, y_min, x_max, y_max])

        return (
            torch.tensor(img).float(),
            torch.tensor(box).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(1).long(),
            self.img_files[index],
            self.gt_files[index]
        )
