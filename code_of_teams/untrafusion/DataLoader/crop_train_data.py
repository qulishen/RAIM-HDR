import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

def crop_scene(scene_path, output_dir, scene, patch_size, stride):
    """处理单个场景，减少内存占用"""
    # 一次性读取所有图片到内存
    img_names = [f for f in os.listdir(scene_path) if f.endswith('.jpg')]
    if not img_names:
        return
    
    # 一次性读取所有图片
    images = {}
    for img_name in img_names:
        img_path = os.path.join(scene_path, img_name)
        images[img_name] = cv2.imread(img_path)
    
    # 使用第一张图确定尺寸
    ref_img = next(iter(images.values()))
    H, W, _ = ref_img.shape
    
    # 计算分块位置
    y_indices = range(0, H - patch_size + 1, stride)
    x_indices = range(0, W - patch_size + 1, stride)
    
    # 预计算所有裁剪位置
    patches = []
    patch_idx = 1
    for y in y_indices:
        for x in x_indices:
            patches.append((y, x, patch_idx))
            patch_idx += 1
    
    # 批量处理每个位置的裁剪
    for y, x, idx in patches:
        new_scene_name = f"{int(scene):03d}{idx:03d}"
        save_path = os.path.join(output_dir, new_scene_name)
        
        # 一次性创建目录（如果不存在）
        os.makedirs(save_path, exist_ok=True)
        
        # 裁剪并保存该位置的所有图片
        for img_name, img in images.items():
            patch = img[y:y + patch_size, x:x + patch_size]
            cv2.imwrite(os.path.join(save_path, img_name), patch)
    
    # 清理内存
    del images

def crop_dataset_optimized(input_dir, output_dir, patch_size, stride, num_workers=None):
    """优化版本：减少IO，支持并行处理"""
    scenes = sorted([d for d in os.listdir(input_dir) 
                     if os.path.isdir(os.path.join(input_dir, d))])
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始处理数据集: {input_dir}")
    print(f"目标路径: {output_dir}")
    print(f"分块大小: {patch_size}x{patch_size}, 步长: {stride}")
    
    # 方法1：顺序处理（内存优化版）
    for scene in tqdm(scenes, desc="总进度"):
        scene_path = os.path.join(input_dir, scene)
        crop_scene(scene_path, output_dir, scene, patch_size, stride)
    
    print("\n所有分块处理完成！")

def crop_dataset_parallel(input_dir, output_dir, patch_size, stride, num_workers=None):
    """并行处理版本（适用于大数据集）"""
    scenes = sorted([d for d in os.listdir(input_dir) 
                     if os.path.isdir(os.path.join(input_dir, d))])
    
    os.makedirs(output_dir, exist_ok=True)
    
    if num_workers is None:
        num_workers = multiprocessing.cpu_count() // 2  # 避免占用全部CPU
    
    # 准备任务参数
    tasks = [(os.path.join(input_dir, scene), output_dir, scene, patch_size, stride) 
             for scene in scenes]
    
    print(f"使用 {num_workers} 个进程并行处理...")
    
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(lambda args: crop_scene(*args), tasks),
            total=len(tasks),
            desc="并行处理进度"
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="优化版数据集离线分块脚本")
    parser.add_argument("--input", type=str, default="./dataset/train/train", help="原始训练集路径")
    parser.add_argument("--output", type=str, default="./dataset/train/crop_size256_stride128", help="分块后数据集存放路径")
    parser.add_argument("--size", type=int, default=256, help="裁剪块的大小")
    parser.add_argument("--stride", type=int, default=128, help="滑动步长")
    parser.add_argument("--parallel", action="store_true", help="启用并行处理")
    parser.add_argument("--workers", type=int, default=4, help="并行工作进程数")
    
    args = parser.parse_args()
    
    if args.parallel:
        crop_dataset_parallel(args.input, args.output, args.size, args.stride, args.workers)
    else:
        crop_dataset_optimized(args.input, args.output, args.size, args.stride)