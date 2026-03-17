import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import random

def crop_scene(scene_path, output_dir, scene, patch_size, stride, jitter_range=0, rotation_range=0, delta_jitter=2, delta_rotation=0.5):
    """
    处理单个场景，支持“全局+帧间微调”的非一致性数据增强
    delta_jitter: 帧与帧之间额外的位移抖动范围 (建议 1-3 像素)
    delta_rotation: 帧与帧之间额外的旋转角度范围 (建议 0.1-0.5 度)
    """
    img_names = sorted([f for f in os.listdir(scene_path) if f.endswith('.jpg')])
    if not img_names:
        return
    
    # 假设中间帧是参考帧（通常是第5张，索引为4）
    ref_idx = len(img_names) // 2
    
    images = {}
    for img_name in img_names:
        img_path = os.path.join(scene_path, img_name)
        images[img_name] = cv2.imread(img_path)
    
    ref_img = list(images.values())[0]
    H, W, _ = ref_img.shape
    
    y_indices = range(0, H - patch_size + 1, stride)
    x_indices = range(0, W - patch_size + 1, stride)
    
    patch_idx = 1
    for y in y_indices:
        for x in x_indices:
            # 1. 这一组 Patch 的全局基准变换
            global_off_y = random.randint(-jitter_range, jitter_range) if jitter_range > 0 else 0
            global_off_x = random.randint(-jitter_range, jitter_range) if jitter_range > 0 else 0
            global_angle = random.uniform(-rotation_range, rotation_range) if rotation_range > 0 else 0
            
            new_scene_name = f"{int(scene):03d}{patch_idx:03d}"
            scene_output_path = os.path.join(output_dir, new_scene_name)
            os.makedirs(scene_output_path, exist_ok=True)
            
            # 2. 遍历每一帧，施加不同的微调
            for i, img_name in enumerate(img_names):
                img = images[img_name]
                
                # 如果是参考帧，增量为 0，保证 GT 对应的参考基准稳定
                if i == ref_idx:
                    d_y, d_x, d_angle = 0, 0, 0
                else:
                    d_y = random.randint(-delta_jitter, delta_jitter)
                    d_x = random.randint(-delta_jitter, delta_jitter)
                    d_angle = random.uniform(-delta_rotation, delta_rotation)
                
                # 最终坐标：基准 + 全局 + 增量
                final_y = max(0, min(H - patch_size, y + global_off_y + d_y))
                final_x = max(0, min(W - patch_size, x + global_off_x + d_x))
                final_angle = global_angle + d_angle
                
                # 执行裁剪
                patch = img[final_y:final_y+patch_size, final_x:final_x+patch_size]
                
                # 执行旋转（带镜像填充）
                if final_angle != 0:
                    center = (patch_size // 2, patch_size // 2)
                    M = cv2.getRotationMatrix2D(center, final_angle, 1.0)
                    patch = cv2.warpAffine(patch, M, (patch_size, patch_size), 
                                         flags=cv2.INTER_LINEAR, 
                                         borderMode=cv2.BORDER_REFLECT)
                
                cv2.imwrite(os.path.join(scene_output_path, img_name), patch)
            
            patch_idx += 1

def main():
    parser = argparse.ArgumentParser(description="非一致性增强数据集裁剪脚本")
    parser.add_argument("--input", type=str, default="/public/home/gs_cs/HDR/dataset/competition/train/train", help="原始训练集路径")
    parser.add_argument("--output", type=str, default="/public/home/gs_cs/HDR/dataset/competition/train/crop_size256_stride128_extra", help="分块后数据集存放路径")
    parser.add_argument("--size", type=int, default=256)#crop_size256_stride128_extra
    parser.add_argument("--stride", type=int, default=128)
    
    # 全局抖动参数
    parser.add_argument("--jitter", type=int, default=8, help="全局抖动范围")
    parser.add_argument("--rotation", type=float, default=3.0, help="全局旋转范围")
    
    # 帧间微调参数（重点）
    parser.add_argument("--d_jitter", type=int, default=2, help="帧间额外抖动像素")
    parser.add_argument("--d_rotation", type=float, default=0.5, help="帧间额外旋转角度")
    
    args = parser.parse_args()
    
    scenes = sorted([d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))])
    os.makedirs(args.output, exist_ok=True)
    
    # 使用进程池
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() // 2) as executor:
        for scene in scenes:
            executor.submit(crop_scene, os.path.join(args.input, scene), args.output, scene, 
                            args.size, args.stride, args.jitter, args.rotation, 
                            args.d_jitter, args.d_rotation)

if __name__ == "__main__":
    main()