from prepare import pair_directories
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class TestDataset(Dataset):
    def __init__(self, data_root, resize_size=(384, 384), label_resize_size=None):
        """
        Args:
            data_root (str): 数据集根目录
            resize_size (tuple): 图像缩放尺寸，可以通过设置较小尺寸实现降采样
        """
        self.data_root = data_root
        self.resize_size = resize_size
        self.label_resize_size = label_resize_size if label_resize_size else resize_size
        pairs = pair_directories(Path(data_root) / "images", Path(data_root) / "masks")
        if not pairs:
            pairs = pair_directories(Path(data_root) / "PNG" / "Original", Path(data_root) / "PNG" / "Ground Truth")
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
        box = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img).float(),
            torch.tensor(box).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(1).long(),
            self.img_files[index],
            self.gt_files[index]
        )
