import os
import re
from glob import glob
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import random
import numpy as np
import torch
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import imutils


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
            exposure_paths = sorted(
                [
                    p for p in glob(os.path.join(folder_path, '*.[jJ][pP][gG]')) +
                               glob(os.path.join(folder_path, '*.png')) +
                               glob(os.path.join(folder_path, '*.jpeg')) +
                               glob(os.path.join(folder_path, '*.JPEG'))
                    if os.path.splitext(os.path.basename(p))[0].lower() != 'hdr'
                ]
            )

            gt_candidates = glob(os.path.join(folder_path, 'HDR.*')) + glob(os.path.join(folder_path, 'hdr.*'))
            gt_candidates = sorted(gt_candidates)
            gt_path = next((p for p in gt_candidates if os.path.isfile(p)), None)

            self.samples.append({'exposures': exposure_paths, 'gt_path': gt_path})

        self.spatial_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # A.RandomResizedCrop(
            #     size=(self.opt['gt_size'],self.opt['gt_size']),
            #     scale=(0.8, 1.0),  # 裁剪原始图像的80%-100%
            #     ratio=(0.8, 1.2),  # 宽高比范围
            #     p=0.5)
        ])

        # self.brightness_transforms = [
        #     A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, -0.1)),
        #     A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.1, 0)),
        #     A.Compose([]),
        #     A.RandomBrightnessContrast(p=0.5, brightness_limit=(0, 0.1)),
        #     A.RandomBrightnessContrast(p=0.5, brightness_limit=(0.1, 0.2)),
        # ]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        sample = self.samples[index % len(self.samples)]

        gt_path = sample['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except Exception:
            raise Exception(f"gt path {gt_path} not working")

        def _extract_index(p):
            name = os.path.basename(p)
            nums = re.findall(r'\d+', name)
            return int(nums[-1]) if nums else 0

        exposures_sorted = sorted(sample['exposures'], key=_extract_index)

        mid_idx = (len(exposures_sorted) - 1) // 2

        def _safe_path(idx):
            if idx < 0:
                idx = len(exposures_sorted) + idx
            idx = max(0, min(idx, len(exposures_sorted) - 1))
            return exposures_sorted[idx]

        selected_left = random.sample(list(range(0, mid_idx)), 2)
        selected_left.sort()
        selected_right = random.sample(list(range(mid_idx + 1, len(exposures_sorted))), 2)
        selected_right.sort()
        order_indices = selected_left + [mid_idx] + selected_right

        ordered_paths = [_safe_path(i) for i in order_indices]

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

        gt_size = self.opt['gt_size']
        h, w, _ = lq_sequence[0].shape
        h_pad = max(0, gt_size - h)
        w_pad = max(0, gt_size - w)
        if h_pad > 0 or w_pad > 0:
            pad_fn = lambda x: cv2.copyMakeBorder(x, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
            img_gt = pad_fn(img_gt)
            lq_sequence = [pad_fn(img) for img in lq_sequence]
        h, w, _ = lq_sequence[0].shape

        if h <= w:
            new_h = random.randint(gt_size, h)
            lq_sequence = [imutils.resize(img, height=new_h) for img in lq_sequence]
            img_gt = imutils.resize(img_gt, height=new_h)
        else:
            new_w = random.randint(gt_size, w)
            lq_sequence = [imutils.resize(img, width=new_w) for img in lq_sequence]
            img_gt = imutils.resize(img_gt, width=new_w)

        # if self.opt['phase'] == 'train':
        #     lq_sequence = [self.brightness_transforms[i](image=d)["image"] for i, d in enumerate(lq_sequence)]

        lq_list = lq_sequence
        img_gt, lq_list = paired_random_crop(img_gt, lq_list, gt_size, scale, gt_path)
        full_view_img = cv2.resize(lq_sequence[2], dsize=(gt_size, gt_size))

        # fig, axs = plt.subplots(2, 7, figsize=(20, 8))
        # fig.suptitle(gt_path)
        # for i in range(5):
        #     axs[0, i].imshow(lq_list[i][:,:,::-1])
        #     axs[0, i].set_title(f'input {i}')
        #     axs[0, i].axis('off')  # 关闭坐标轴
        # axs[0, 5].imshow(img_gt[:,:,::-1])
        # axs[0, 5].set_title(f'HDR')
        # axs[0, 5].axis('off')  # 关闭坐标轴
        # axs[0, 6].imshow(full_view_img[:,:,::-1])
        # axs[0, 6].set_title(f'full_view_img')
        # axs[0, 6].axis('off')  # 关闭坐标轴

        aug_img = np.concatenate([img_gt] + lq_list + [full_view_img], axis=-1)
        aug_img = self.spatial_transforms(image=aug_img)["image"]
        aug_img = np.array_split(aug_img, axis=-1, indices_or_sections=7)
        img_gt, lq_list, full_view_img = aug_img[0], aug_img[1:-1], aug_img[-1]

        #
        # for i in range(5):
        #     axs[1, i].imshow(lq_list[i][:,:,::-1])
        #     axs[1, i].set_title(f'input {i}')
        #     axs[1, i].axis('off')  # 关闭坐标轴
        # axs[1, 5].imshow(img_gt[:,:,::-1])
        # axs[1, 5].set_title(f'HDR')
        # axs[1, 5].axis('off')  # 关闭坐标轴
        # axs[1, 6].imshow(full_view_img[:,:,::-1])
        # axs[1, 6].set_title(f'full_view_img')
        # axs[1, 6].axis('off')  # 关闭坐标轴
        # plt.tight_layout()
        # plt.show()

        tensors = img2tensor([img_gt] + lq_list + [full_view_img], bgr2rgb=True, float32=True)
        img_gt = tensors[0]
        lq_tensors = tensors[1:]  # full_view_img = tensors[-1]

        if self.mean is not None or self.std is not None:
            normalize(img_gt, self.mean, self.std, inplace=True)
            for t in lq_tensors:
                normalize(t, self.mean, self.std, inplace=True)

        img_lq = torch.cat(lq_tensors, dim=0)
        return {
            'lq': img_lq,
            'lq_path': gt_path,
            'gt': img_gt
        }

    def __len__(self):
        return len(self.samples)


