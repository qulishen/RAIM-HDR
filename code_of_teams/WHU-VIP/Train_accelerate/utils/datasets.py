# -*- coding: utf-8 -*-
"""Training dataset and data augmentation utilities. — Team: whu-vip"""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RepeatDataset(Dataset):
    """Wraps a dataset and repeats it N times to increase effective epoch length."""
    def __init__(self, dataset, repeats=1):
        self.dataset = dataset
        self.repeats = repeats

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


class RAIMTrainDataset(Dataset):
    """Dataset loader for NTIRE 2026 RAIM Track2 training patches.

    Expects pre-cropped .npy patches under:
      data_root/inputs/  — (H, W, 9) uint8, stacked 3-frame LDR
      data_root/gt/      — (H, W, 3) uint8, HDR ground truth

    Args:
        data_root (str)      : Root directory containing inputs/ and gt/
        crop_size (int)      : Random crop size applied during loading
        use_strong_aug (bool): Enable strong augmentation pool (60% trigger rate)
    """

    def __init__(self, data_root, crop_size=256, use_strong_aug=False):
        self.data_root      = data_root
        self.crop_size      = crop_size
        self.use_strong_aug = use_strong_aug
        self.inputs_dir     = os.path.join(data_root, 'inputs')
        self.gt_dir         = os.path.join(data_root, 'gt')
        self.file_names     = sorted(os.listdir(self.inputs_dir))
        self.aug_pool       = [
            self._aug_channel_shuffle,
            self._aug_crf_perturbation,
            self._aug_misalignment,
            self._aug_local_exp,
        ]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name   = self.file_names[idx]
        inputs = np.load(os.path.join(self.inputs_dir, name)).astype(np.float32)
        gt     = np.load(os.path.join(self.gt_dir,     name)).astype(np.float32)

        # Random crop
        h, w, _ = inputs.shape
        if h > self.crop_size and w > self.crop_size:
            top  = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            inputs = inputs[top:top + self.crop_size, left:left + self.crop_size]
            gt     = gt    [top:top + self.crop_size, left:left + self.crop_size]
        else:
            inputs = cv2.resize(inputs, (self.crop_size, self.crop_size))
            gt     = cv2.resize(gt,     (self.crop_size, self.crop_size))

        # Basic geometric augmentation
        inputs, gt = self._random_flip_rotate(inputs, gt)

        # Strong augmentation pool (optional, 60% trigger)
        if self.use_strong_aug and random.random() < 0.6:
            for aug in random.sample(self.aug_pool, random.choice([1, 2])):
                inputs, gt = aug(inputs, gt)

        # Ensure contiguous memory layout before converting to tensor
        inputs_t = torch.from_numpy(np.ascontiguousarray(inputs.transpose(2, 0, 1)))
        gt_t     = torch.from_numpy(np.ascontiguousarray(gt.transpose(2, 0, 1)))
        return inputs_t, gt_t

    # -------------------------------------------------------------------------
    # Basic augmentation
    # -------------------------------------------------------------------------

    def _random_flip_rotate(self, inputs, gt):
        if random.random() < 0.5:
            inputs, gt = np.flip(inputs, 1), np.flip(gt, 1)
        if random.random() < 0.5:
            inputs, gt = np.flip(inputs, 0), np.flip(gt, 0)
        if random.random() < 0.5:
            k = random.randint(1, 3)
            inputs = np.rot90(inputs, k, axes=(0, 1))
            gt     = np.rot90(gt,     k, axes=(0, 1))
        return inputs, gt

    # -------------------------------------------------------------------------
    # Strong augmentation pool
    # -------------------------------------------------------------------------

    def _aug_channel_shuffle(self, inputs, gt):
        """Randomly permute RGB channels consistently across all frames and GT."""
        perm = np.random.permutation(3)
        inputs[:, :, 0:3] = inputs[:, :, perm]
        inputs[:, :, 3:6] = inputs[:, :, 3 + perm]
        inputs[:, :, 6:9] = inputs[:, :, 6 + perm]
        gt = gt[:, :, perm]
        return inputs, gt

    def _aug_crf_perturbation(self, inputs, gt):
        """Simulate camera response curve variation via random gamma correction."""
        gamma  = random.uniform(0.9, 1.1)
        inputs = np.clip(inputs, 0, 1) ** gamma
        return inputs, gt

    def _aug_misalignment(self, inputs, gt):
        """Simulate small spatial misalignment on under/over-exposed frames."""
        max_shift = 4
        h, w      = inputs.shape[:2]
        for fi in [0, 6]:
            sx = random.randint(-max_shift, max_shift)
            sy = random.randint(-max_shift, max_shift)
            M  = np.float32([[1, 0, sx], [0, 1, sy]])
            inputs[:, :, fi:fi + 3] = cv2.warpAffine(
                inputs[:, :, fi:fi + 3], M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return inputs, gt

    def _aug_local_exp(self, inputs, gt):
        """Apply random Gaussian local exposure perturbation to non-reference frames."""
        h, w = inputs.shape[:2]
        for fi in [0, 6]:
            if random.random() < 0.5:
                cx, cy = random.randint(0, w), random.randint(0, h)
                sigma  = random.uniform(20, 80)
                y, x   = np.ogrid[-cy:h - cy, -cx:w - cx]
                mask   = np.exp(-(x * x + y * y) / (2 * sigma * sigma))[..., np.newaxis]
                factor = random.uniform(0.5, 1.5)
                frame  = inputs[:, :, fi:fi + 3]
                inputs[:, :, fi:fi + 3] = np.clip(frame * (1 - mask) + frame * factor * mask, 0, 1)
        return inputs, gt