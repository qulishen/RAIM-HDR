# -*- coding: utf-8 -*-
"""
Training Dataset Preparation Script — NTIRE 2026 RAIM Track2 (HDR Reconstruction)
Team: whu-vip

Description:
    Reads 3 LDR images per scene (under-/normal-/over-exposed) and the HDR ground truth,
    crops them into overlapping patches, and saves them as .npy files for efficient training.

    Input  : 3 RGB JPEG images per scene -> stacked to a 9-channel array (H, W, 9) uint8
    Output : inputs/*.npy  (9-channel patches)
             gt/*.npy      (3-channel GT patches)

    Training image filenames  : 1.jpg (under), 3.jpg (normal), 5.jpg (over)
    Inference image filenames : 0.jpg (under), 2.jpg (normal), 4.jpg (over)
"""

import os
import cv2
import numpy as np
import glob
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ================= Default Configuration =================
DEFAULT_SOURCE_DIR  = './Datasets/train'
DEFAULT_TARGET_DIR  = './Train_accelerate/datas'
DEFAULT_PATCH_SIZE  = 512
DEFAULT_STRIDE      = 256
DEFAULT_NUM_WORKERS = max(1, cpu_count() - 4)

LDR_NAMES = ['1.jpg', '3.jpg', '5.jpg']  # under / normal / over
GT_NAME   = 'HDR.jpg'

# Scenes excluded from training due to quality issues (fixed, do not modify)
BLACKLIST = {
    '089', '091', '090', '093', '061',
    '092', '060', '095', '094', '003',
}
# =========================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description='NTIRE 2026 RAIM Track2 — Training Dataset Preparation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--source_dir',  type=str, default=DEFAULT_SOURCE_DIR,
                        help='Root directory of raw training scenes')
    parser.add_argument('--target_dir',  type=str, default=DEFAULT_TARGET_DIR,
                        help='Output directory (will contain inputs/ and gt/ sub-dirs)')
    parser.add_argument('--patch_size',  type=int, default=DEFAULT_PATCH_SIZE,
                        help='Crop patch size (pixels)')
    parser.add_argument('--stride',      type=int, default=DEFAULT_STRIDE,
                        help='Sliding-window stride for patch extraction')
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS,
                        help='Number of parallel worker processes')
    return parser.parse_args()


def get_crop_coords(H, W, patch_size, stride):
    """
    Generate top-left (y, x) coordinates for all patches using a sliding window.
    Ensures the last patch always covers the image boundary (no data loss at edges).
    """
    h_steps = list(range(0, H - patch_size + 1, stride))
    if not h_steps or h_steps[-1] != H - patch_size:
        h_steps.append(H - patch_size)

    w_steps = list(range(0, W - patch_size + 1, stride))
    if not w_steps or w_steps[-1] != W - patch_size:
        w_steps.append(W - patch_size)

    return [(h, w) for h in h_steps for w in w_steps]


def process_scene(args_tuple):
    """
    Worker function: processes one scene and saves all patches.
    Args are packed into a tuple to safely bypass multiprocessing scope issues.
    """
    scene_path, patch_size, stride, target_dir = args_tuple
    scene_id = os.path.basename(scene_path)

    # ------ 1. Load and stack 9-channel LDR input ------
    frames = []
    for fname in LDR_NAMES:
        img_path = os.path.join(scene_path, fname)
        if not os.path.exists(img_path):
            print(f"[Warning] Missing {fname} in scene {scene_id}, skipping.")
            return
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Warning] Failed to read {img_path}, skipping.")
            return
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Channel order: [R1,G1,B1, R3,G3,B3, R5,G5,B5]
    input_volume = np.concatenate(frames, axis=2).astype(np.uint8)

    # ------ 2. Load GT ------
    gt_path = os.path.join(scene_path, GT_NAME)
    if not os.path.exists(gt_path):
        print(f"[Warning] Missing GT in scene {scene_id}, skipping.")
        return
    gt_img = cv2.imread(gt_path)
    if gt_img is None:
        print(f"[Warning] Failed to read GT in scene {scene_id}, skipping.")
        return
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB).astype(np.uint8)

    # ------ 3. Sanity checks ------
    H, W = input_volume.shape[:2]
    if (H, W) != gt_img.shape[:2]:
        print(f"[Skip] Dimension mismatch between LDR and GT in scene {scene_id}.")
        return
    if H < patch_size or W < patch_size:
        print(f"[Skip] Scene {scene_id} is smaller than patch size ({patch_size}px), skipping.")
        return

    # ------ 4. Crop and save patches ------
    coords = get_crop_coords(H, W, patch_size, stride)
    for idx, (y, x) in enumerate(coords):
        patch_input = input_volume[y:y + patch_size, x:x + patch_size, :]
        patch_gt    = gt_img[y:y + patch_size, x:x + patch_size, :]

        save_name = f"{scene_id}_{idx:04d}_y{y}_x{x}.npy"
        np.save(os.path.join(target_dir, 'inputs', save_name), patch_input)
        np.save(os.path.join(target_dir, 'gt',     save_name), patch_gt)


def main():
    args = parse_args()

    if not os.path.exists(args.source_dir):
        print(f"[Error] Source directory not found: {args.source_dir}")
        return

    os.makedirs(os.path.join(args.target_dir, 'inputs'), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, 'gt'),     exist_ok=True)

    all_scene_paths = sorted([
        p for p in glob.glob(os.path.join(args.source_dir, '*'))
        if os.path.isdir(p)
    ])

    scene_paths = [p for p in all_scene_paths if os.path.basename(p) not in BLACKLIST]
    skipped     = len(all_scene_paths) - len(scene_paths)

    print("=" * 52)
    print(f"  Task           : LDR (1,3,5.jpg) -> 9-ch Input")
    print(f"  Source         : {args.source_dir}")
    print(f"  Target         : {args.target_dir}")
    print(f"  Patch size     : {args.patch_size}  Stride: {args.stride}")
    print(f"  Workers        : {args.num_workers}")
    print(f"  Total scenes   : {len(all_scene_paths)}")
    print(f"  Blacklisted    : {skipped}")
    print(f"  To process     : {len(scene_paths)}")
    print("=" * 52)

    if not scene_paths:
        print("No scenes to process. Exiting.")
        return

    # Pack arguments for robust multiprocessing
    task_args = [(path, args.patch_size, args.stride, args.target_dir) for path in scene_paths]

    with Pool(processes=args.num_workers) as pool:
        list(tqdm(pool.imap(process_scene, task_args), total=len(task_args), desc="Processing scenes"))

    print("\nDataset preparation complete.")
    print(f"  inputs/ -> {os.path.join(args.target_dir, 'inputs')}")
    print(f"  gt/     -> {os.path.join(args.target_dir, 'gt')}")


if __name__ == '__main__':
    main()