"""
Dataset classes for Multi-Exposure Fusion (MEF) training and inference.
Train: 100 sequences x 7 exposures (0.jpg-6.jpg) + HDR.jpg GT
Val:   100 sequences x 5 exposures (0.jpg-4.jpg), no GT
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import random


class TrainSet_MEF(Dataset):
    """Training dataset for multi-exposure fusion.

    Each scene has 7 exposures (0.jpg-6.jpg) and one GT (HDR.jpg).
    We select 5 evenly-spaced exposures (0, 1, 3, 4, 6) to match
    the 5-exposure format of the validation set.
    Input: 5 x 3 = 15 channels (concatenated RGB).
    Output: 3 channels (HDR.jpg).
    """

    def __init__(self, args):
        super().__init__()
        self.root = args.traindata_root
        self.patch_size = getattr(args, 'patch_size', 256)
        self.phase = getattr(args, 'phase', 'train')
        self.random_exposure_subset = getattr(args, 'random_exposure_subset', False)
        self.keep_extreme_exposures = getattr(args, 'keep_extreme_exposures', False)

        # Select 5 from 7 exposures: evenly spaced
        self.frame_indices = getattr(args, 'frame_indices', [0, 1, 3, 4, 6])
        self.num_frames = len(self.frame_indices)

        # Scan for scenes
        self.scenes = self._scan_scenes()

        # Train/val split
        self.val_fraction = getattr(args, 'val_scene_fraction', 0.1)
        self._split_scenes()

        # Subset sampling
        self.subset_fraction = getattr(args, 'train_subset_fraction', 1.0)
        self.epoch = 0
        self._create_epoch_mapping()

    def _scan_scenes(self):
        """Scan for scene directories containing exposure images and GT."""
        scenes = []
        if not os.path.isdir(self.root):
            return scenes
        for scene_name in sorted(os.listdir(self.root)):
            scene_dir = os.path.join(self.root, scene_name)
            if not os.path.isdir(scene_dir):
                continue
            gt_path = os.path.join(scene_dir, 'HDR.jpg')
            if not os.path.exists(gt_path):
                continue
            # Check that all required exposure files exist
            all_exist = True
            for idx in self.frame_indices:
                if not os.path.exists(os.path.join(scene_dir, f'{idx}.jpg')):
                    all_exist = False
                    break
            if all_exist:
                scenes.append(scene_name)
        return scenes

    def _split_scenes(self):
        n_val = max(1, int(len(self.scenes) * self.val_fraction))
        if self.phase == 'train':
            self.scenes = self.scenes[:-n_val] if n_val > 0 else self.scenes
        else:
            self.scenes = self.scenes[-n_val:] if n_val > 0 else []

    def _create_epoch_mapping(self):
        random.seed(42 + self.epoch)
        if self.subset_fraction < 1.0:
            self.epoch_size = max(1, int(len(self.scenes) * self.subset_fraction))
            self.epoch_indices = random.sample(range(len(self.scenes)), self.epoch_size)
        else:
            self.epoch_size = len(self.scenes)
            self.epoch_indices = list(range(len(self.scenes)))
        random.shuffle(self.epoch_indices)

    def set_epoch(self, epoch):
        if epoch != self.epoch:
            self.epoch = epoch
            self._create_epoch_mapping()

    def __len__(self):
        return self.epoch_size

    def _sample_frame_indices(self, scene_dir):
        available = sorted([
            int(f.split('.')[0]) for f in os.listdir(scene_dir)
            if f.endswith('.jpg') and f[0].isdigit()
        ])
        if len(available) <= self.num_frames or not self.random_exposure_subset:
            return list(self.frame_indices)

        if self.keep_extreme_exposures and len(available) >= 2:
            anchors = [available[0], available[-1]]
            middle = available[1:-1]
            sample_count = max(0, self.num_frames - len(anchors))
            picked = random.sample(middle, sample_count) if len(middle) >= sample_count else middle
            return sorted(anchors + picked)

        return sorted(random.sample(available, self.num_frames))

    def __getitem__(self, idx):
        if idx == 0:
            self.epoch += 1
            self._create_epoch_mapping()

        scene_idx = self.epoch_indices[idx]
        scene_name = self.scenes[scene_idx]
        scene_dir = os.path.join(self.root, scene_name)
        frame_indices = self._sample_frame_indices(scene_dir)

        # Load selected exposure frames
        frames = []
        for fidx in frame_indices:
            img = cv2.imread(os.path.join(scene_dir, f'{fidx}.jpg'), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

        # Load GT
        gt = cv2.imread(os.path.join(scene_dir, 'HDR.jpg'), cv2.IMREAD_COLOR)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        h, w, _ = gt.shape

        # Random crop
        if h > self.patch_size and w > self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            frames = [f[top:top+self.patch_size, left:left+self.patch_size] for f in frames]
            gt = gt[top:top+self.patch_size, left:left+self.patch_size]

        # Stack frames: [5*3, H, W]
        input_stack = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)
        gt = gt.transpose(2, 0, 1)  # [3, H, W]

        # Augmentation
        if random.random() < 0.5:
            input_stack = input_stack[:, :, ::-1]
            gt = gt[:, :, ::-1]
        if random.random() < 0.5:
            input_stack = input_stack[:, ::-1, :]
            gt = gt[:, ::-1, :]
        if random.random() < 0.5:
            input_stack = input_stack.transpose(0, 2, 1)
            gt = gt.transpose(0, 2, 1)

        # To tensor, normalize to [0, 1]
        input_t = torch.from_numpy(
            np.ascontiguousarray(input_stack).astype(np.float32)) / 255.0
        gt_t = torch.from_numpy(
            np.ascontiguousarray(gt).astype(np.float32)) / 255.0

        return {'images': input_t, 'labels': gt_t, 'name': scene_name}


class TestSet_MEF(Dataset):
    """Validation/test dataset for multi-exposure fusion.

    Each scene has 5 exposures (0.jpg-4.jpg), no GT.
    For internal validation (from train split), GT (HDR.jpg) is loaded if available.
    """

    def __init__(self, args):
        super().__init__()
        self.root = args.traindata_root
        self.target_size = getattr(args, 'target_size', 512)
        self.phase = 'valid'
        self.num_frames = 5  # Val always has 5 frames

        # Scan for scenes
        all_scenes = self._scan_scenes()

        # Use validation split
        val_fraction = getattr(args, 'val_scene_fraction', 0.1)
        n_val = max(1, int(len(all_scenes) * val_fraction))
        self.scenes = all_scenes[-n_val:] if n_val > 0 else []
        self.scenes = self.scenes[:50]

    def _scan_scenes(self):
        scenes = []
        if not os.path.isdir(self.root):
            return scenes
        for scene_name in sorted(os.listdir(self.root)):
            scene_dir = os.path.join(self.root, scene_name)
            if not os.path.isdir(scene_dir):
                continue
            # Need at least 5 exposure frames
            if os.path.exists(os.path.join(scene_dir, '0.jpg')):
                scenes.append(scene_name)
        return scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_name = self.scenes[idx]
        scene_dir = os.path.join(self.root, scene_name)

        # Load 5 exposure frames (indices 0-4)
        # For train scenes (7 exposures), select 5 evenly: [0,1,3,4,6]
        # For val scenes (5 exposures), use all: [0,1,2,3,4]
        available = sorted([int(f.split('.')[0]) for f in os.listdir(scene_dir)
                          if f.endswith('.jpg') and f[0].isdigit()])
        if len(available) == 7:
            frame_indices = [0, 1, 3, 4, 6]
        else:
            frame_indices = available[:5]

        frames = []
        for fidx in frame_indices:
            img = cv2.imread(os.path.join(scene_dir, f'{fidx}.jpg'), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

        h, w, _ = frames[0].shape

        # Load GT if available
        gt_path = os.path.join(scene_dir, 'HDR.jpg')
        has_gt = os.path.exists(gt_path)
        if has_gt:
            gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        # Center crop
        if h > self.target_size and w > self.target_size:
            top = (h - self.target_size) // 2
            left = (w - self.target_size) // 2
            frames = [f[top:top+self.target_size, left:left+self.target_size] for f in frames]
            if has_gt:
                gt = gt[top:top+self.target_size, left:left+self.target_size]

        # Stack frames: [5*3, H, W]
        input_stack = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)
        input_t = torch.from_numpy(
            np.ascontiguousarray(input_stack).astype(np.float32)) / 255.0

        if has_gt:
            gt = gt.transpose(2, 0, 1)
            gt_t = torch.from_numpy(
                np.ascontiguousarray(gt).astype(np.float32)) / 255.0
        else:
            gt_t = torch.zeros(3, input_t.shape[1], input_t.shape[2])

        return {'images': input_t, 'labels': gt_t, 'name': scene_name}
