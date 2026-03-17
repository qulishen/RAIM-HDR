import os
import random
import numpy as np
import cv2
from typing import List, Dict, Tuple
from torch.utils.data import Dataset


# 假设 utils 中有这两个函数，如果没有请自行实现
# from utils import hwc_to_chw, read_img

def read_img(path: str) -> np.ndarray:
    """读取图像并归一化到 [0, 1]，请确保你的 utils.read_img 是这样实现的"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image not found or corrupted: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def hwc_to_chw(img: np.ndarray) -> np.ndarray:
    """ [H, W, C] -> [C, H, W] """
    return np.transpose(img, (2, 0, 1))


def augment(imgs: List[np.ndarray], size: int = 256, edge_decay: float = 0., only_h_flip: bool = False) -> List[
    np.ndarray]:
    """对一组图像进行同步的数据增强（Random Crop, Flip, Rotate）"""
    H, W, _ = imgs[0].shape
    Hc, Wc = size, size

    # 1. Random Crop (同步裁剪)
    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H - Hc)

    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W - Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    # 2. Horizontal Flip (水平翻转)
    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            # [修复这里] 加上 .copy() 消除负步长
            imgs[i] = np.flip(imgs[i], axis=1).copy()

    # 3. Rotation (旋转)
    if not only_h_flip:
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            # [修复这里] 加上 .copy() 消除负步长
            imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1)).copy()

    return imgs


def align(imgs: List[np.ndarray], size: int = 256) -> List[np.ndarray]:
    """验证集使用的 Center Crop"""
    H, W, _ = imgs[0].shape
    Hc, Wc = size, size

    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    return imgs


class HDRDataset(Dataset):
    def __init__(self, data_dir: str, mode: str, size: int = 256, edge_decay: float = 0., only_h_flip: bool = False):
        """
        HDR 数据集读取类
        Args:
            data_dir: 数据集根目录，例如 './data/HDR'
            mode: 'train', 'valid', 或 'test'
            size: 裁剪尺寸
        """
        assert mode in ['train', 'valid', 'test'], "Mode must be train, valid, or test."

        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip

        # 拼接路径，例如 ./data/HDR/train
        self.split_dir = os.path.join(data_dir, mode)

        # 获取所有样本文件夹 (001, 002, ...)
        # 防御性编程：只保留真正的文件夹，排除可能存在的隐藏文件如 .DS_Store
        sample_dirs = sorted([
            d for d in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, d))
        ])

        self.valid_samples = []
        self.input_names = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']
        self.gt_name = 'HDR.jpg'

        # 数据校验 (Sanity Check)：确保每个文件夹里都有我们需要的文件
        for sample_name in sample_dirs:
            sample_path = os.path.join(self.split_dir, sample_name)
            is_valid = True

            # 检查 1~5.jpg 和 HDR.jpg 是否都存在
            for img_name in self.input_names + [self.gt_name]:
                if not os.path.exists(os.path.join(sample_path, img_name)):
                    print(f"[Warning] Missing {img_name} in {sample_path}, skipping this sample.")
                    is_valid = False
                    break

            if is_valid:
                self.valid_samples.append(sample_path)

        print(f"[{mode.upper()}] Successfully loaded {len(self.valid_samples)} valid samples.")

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # 禁用 OpenCV 的多线程，防止在 PyTorch DataLoader 多进程下死锁
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        sample_path = self.valid_samples[idx]

        # 1. 读取 5 张输入图像和 1 张 GT
        # 原代码将 [0, 1] 映射到了 [-1, 1]，这里保留这个逻辑
        input_imgs = []
        for img_name in self.input_names:
            img = read_img(os.path.join(sample_path, img_name)) * 2 - 1
            input_imgs.append(img)

        gt_img = read_img(os.path.join(sample_path, self.gt_name)) * 2 - 1

        # 2. 数据增强 (将所有输入和GT放在一个List里同步增强，保证空间对齐)
        all_imgs = input_imgs + [gt_img]

        if self.mode == 'train':
            all_imgs = augment(all_imgs, self.size, self.edge_decay, self.only_h_flip)
        elif self.mode == 'valid':
            all_imgs = align(all_imgs, self.size)

        # 3. 分离 Input 和 GT
        input_imgs = all_imgs[:-1]  # 前5张是输入
        gt_img = all_imgs[-1]  # 最后1张是GT

        # 4. 维度转换 [H, W, C] -> [C, H, W]
        input_tensors = [hwc_to_chw(img) for img in input_imgs]
        gt_tensor = hwc_to_chw(gt_img)

        # 5. 融合 (Concatenation)
        # 将 5 个 [3, H, W] 的张量在通道维度拼接，变成 [15, H, W]
        # 这样网络的第一层卷积 (in_channels=15) 就能同时看到所有曝光的信息
        source_tensor = np.concatenate(input_tensors, axis=0)

        # 如果你真的非常确定要在数据读取阶段就融合成3通道（比如取平均），
        # 请注释掉上面那行，取消下面这行的注释：
        # source_tensor = np.mean(np.stack(input_tensors, axis=0), axis=0)

        return {
            'source': source_tensor,  # Shape: [15, H, W]
            'target': gt_tensor,  # Shape: [3, H, W]
            'sample_name': os.path.basename(sample_path)  # 例如 '001'
        }


class HDRTestDataset(Dataset):
    def __init__(self, data_dir: str):
        """
        专门用于比赛最终测试集的数据读取类 (无 GT, 原分辨率)
        Args:
            data_dir: 测试集根目录，例如 './data/HDR/test'
        """
        self.data_dir = data_dir

        # 获取所有样本文件夹 (090, 091, ...)
        sample_dirs = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ])

        self.valid_samples = []
        # [修改点 1] 文件名改为 0.jpg 到 4.jpg
        self.input_names = ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg']

        # 数据校验
        for sample_name in sample_dirs:
            sample_path = os.path.join(self.data_dir, sample_name)
            is_valid = True

            for img_name in self.input_names:
                if not os.path.exists(os.path.join(sample_path, img_name)):
                    print(f"[Warning] Missing {img_name} in {sample_path}, skipping.")
                    is_valid = False
                    break

            if is_valid:
                self.valid_samples.append(sample_path)

        print(f"[INFERENCE] Successfully loaded {len(self.valid_samples)} test samples.")

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        sample_path = self.valid_samples[idx]

        # 1. 读取 5 张输入图像
        input_imgs = []
        for img_name in self.input_names:
            img = read_img(os.path.join(sample_path, img_name)) * 2 - 1
            input_imgs.append(img)

        # [修改点 2] 不做任何 augment 或 align，保持原图尺寸

        # 2. 维度转换 [H, W, C] -> [C, H, W]
        input_tensors = [hwc_to_chw(img) for img in input_imgs]

        # 3. 融合 (Concatenation) -> [15, H, W]
        source_tensor = np.concatenate(input_tensors, axis=0)

        # [修改点 3] 伪造一个全黑的 Dummy Target，防止 test_step 报错
        # 保持和输入图像相同的空间分辨率 (H, W)
        _, H, W = source_tensor.shape
        dummy_target = np.zeros((3, H, W), dtype=np.float32)

        return {
            'source': source_tensor,
            'target': dummy_target,  # 假的 GT，仅用于占位
            'sample_name': os.path.basename(sample_path)  # 例如 '090'
        }
