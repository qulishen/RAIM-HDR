import os
import re
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

# 【新增】引入 augment 用于几何增强
from basicsr.data.transforms import paired_random_crop, augment
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class HDR_Dataset(data.Dataset):

    def __init__(self, opt):
        super(HDR_Dataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.hdr_root = opt.get('dataroot', opt.get('dataroot_hdr'))

        subfolders = [
            d for d in sorted(os.listdir(self.hdr_root))
            if os.path.isdir(os.path.join(self.hdr_root, d))
        ]

        self.samples = []
        for folder in subfolders:
            folder_path = os.path.join(self.hdr_root, folder)

            # 找到所有非 HDR 命名的图片 (输入序列)
            exposure_paths = sorted([
                p for p in glob(os.path.join(folder_path, '*.[jJ][pP][gG]')) +
                           glob(os.path.join(folder_path, '*.png')) +
                           glob(os.path.join(folder_path, '*.jpeg')) +
                           glob(os.path.join(folder_path, '*.JPEG'))
                if os.path.splitext(os.path.basename(p))[0].lower() != 'hdr'
            ])

            # 找到 GT (HDR 图像)
            gt_candidates = glob(os.path.join(folder_path, 'HDR.*')) + glob(os.path.join(folder_path, 'hdr.*'))
            gt_candidates = sorted(gt_candidates)
            gt_path = next((p for p in gt_candidates if os.path.isfile(p)), None)

            self.samples.append({'exposures': exposure_paths, 'gt_path': gt_path})

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt.get('scale', 1)
        sample = self.samples[index % len(self.samples)]

        # 1. 读取 GT 图像
        gt_path = sample['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except Exception:
            raise Exception(f"gt path {gt_path} not working")

        # 2. 曝光图像排序并提取 -2 ~ +2 范围
        def _extract_index(p):
            name = os.path.basename(p)
            nums = re.findall(r'\d+', name)
            return int(nums[-1]) if nums else 0

        exposures_sorted = sorted(sample['exposures'], key=_extract_index)

        # 动态获取中间 5 张图 (-2, -1, 0, +1, +2)
        mid_idx = len(exposures_sorted) // 2
        order_indices = [mid_idx - 2, mid_idx - 1, mid_idx, mid_idx + 1, mid_idx + 2]

        def _safe_path(idx):
            idx = max(0, min(idx, len(exposures_sorted) - 1))
            return exposures_sorted[idx]

        ordered_paths = [_safe_path(i) for i in order_indices]

        # 3. 读取 LQ 序列
        def _load_img(p):
            try:
                img_bytes = self.file_client.get(p, 'hdr')
                img = imfrombytes(img_bytes, float32=True)
                if img is None:
                    raise ValueError
                return img
            except Exception:
                raise Exception(f"ldr path {p} not working")

        lq_sequence = [_load_img(p) for p in ordered_paths]

        # 4. 边缘 Padding (如果原图比 gt_size 小)
        gt_size = self.opt['gt_size']
        h, w, _ = lq_sequence[0].shape
        h_pad = max(0, gt_size - h)
        w_pad = max(0, gt_size - w)
        if h_pad > 0 or w_pad > 0:
            pad_fn = lambda x: cv2.copyMakeBorder(x, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
            img_gt = pad_fn(img_gt)
            lq_sequence = [pad_fn(img) for img in lq_sequence]

        # 5. 随机裁剪 (Random Crop)
        lq_list = lq_sequence
        img_gt, lq_list = paired_random_crop(img_gt, lq_list, gt_size, scale, gt_path)
        if not isinstance(lq_list, list):
            lq_list = [lq_list]

        #几何数据增强 (翻转 + 旋转)
        if self.opt.get('geometric_augs', False):
            # augment 接收一个列表，会对列表内所有的图片做完全相同的随机翻转和旋转，保证严格对齐
            aug_results = augment([img_gt] + lq_list, hflip=True, rotation=True)
            img_gt = aug_results[0]
            lq_list = aug_results[1:]

        # 7. 转为 Tensor
        tensors = img2tensor([img_gt] + lq_list, bgr2rgb=True, float32=True)
        img_gt = tensors[0]
        lq_tensors = tensors[1:]

        # 8. 归一化 (如果 yaml 中配了 mean 和 std)
        if self.mean is not None or self.std is not None:
            normalize(img_gt, self.mean, self.std, inplace=True)
            for t in lq_tensors:
                normalize(t, self.mean, self.std, inplace=True)

        # 9. 将 5 张 [3, H, W] 拼接成 1 个 [15, H, W] 的 Tensor
        img_lq = torch.cat(lq_tensors, dim=0)

        return {
            'lq': img_lq,
            'lq_path': gt_path,
            'gt': img_gt
        }

    def __len__(self):
        return len(self.samples)