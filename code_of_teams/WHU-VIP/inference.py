# ============================================================
# NTIRE 2026 - RAIM Track2: HDR Reconstruction
# Inference Script for AFUNet (with SFT and GatedMap)
# ============================================================

import os
import cv2
import torch
import argparse
import numpy as np
import time
import zipfile
import torch.nn.functional as F
from torch.cuda.amp import autocast
import multiprocessing as mp
from math import ceil

from Train_accelerate.models.AFUNet_GM_SFT import AFUNet

# --------------- Model Configuration ---------------
MODEL_CONFIG = {}
INPUT_INDICES = [0, 2, 4]

def parse_args():
    parser = argparse.ArgumentParser(description="AFUNet Inference for NTIRE 2026 RAIM HDR Track")
    parser.add_argument('--input_root', type=str, required=True,
                        help='Root directory of test scenes (each sub-folder is a scene)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint (.pth)')
    parser.add_argument('--save_root', type=str, required=True,
                        help='Directory to save output images and submission zip')
    parser.add_argument('--gpu_list', type=int, nargs='+', default=[0],
                        help='List of GPU IDs to use, e.g. --gpu_list 0 1 2')
    parser.add_argument('--whole_image', action='store_true', default=True,
                        help='Use whole-image inference (default: True, fallback to split on OOM)')
    parser.add_argument('--zip_name', type=str, default='submission.zip',
                        help='Name of the output submission zip file')
    return parser.parse_args()

def get_adaptive_mask(size_h, size_w, blend_h, blend_w, fade_sides, device):
    mask = torch.ones((size_h, size_w), dtype=torch.float32, device=device)
    fade_top, fade_bottom, fade_left, fade_right = fade_sides

    if fade_top:
        ramp = torch.linspace(0, 1, blend_h, device=device).view(-1, 1)
        mask[:blend_h, :] *= ramp
    if fade_bottom:
        ramp = torch.linspace(0, 1, blend_h, device=device).flip(0).view(-1, 1)
        mask[-blend_h:, :] *= ramp
    if fade_left:
        ramp = torch.linspace(0, 1, blend_w, device=device).view(1, -1)
        mask[:, :blend_w] *= ramp
    if fade_right:
        ramp = torch.linspace(0, 1, blend_w, device=device).flip(0).view(1, -1)
        mask[:, -blend_w:] *= ramp

    return mask.unsqueeze(0).unsqueeze(0)

def process_input_data(scene_path):
    processed_channels = []
    for idx in INPUT_INDICES:
        img_path = os.path.join(scene_path, f"{idx}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        processed_channels.append(torch.from_numpy(img.transpose(2, 0, 1)))

    return torch.cat(processed_channels, dim=0).unsqueeze(0)

def inference_whole(model, input_tensor):
    b, c, h, w = input_tensor.shape
    align = 128
    pad_h = int(np.ceil(h / align) * align) - h
    pad_w = int(np.ceil(w / align) * align) - w

    img_padded = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect') if (pad_h > 0 or pad_w > 0) else input_tensor

    with autocast():
        out = model(img_padded)
        if isinstance(out, (list, tuple)):
            out = out[0]

    return out[:, :, :h, :w].float()

def inference_split2(model, input_tensor, device):
    b, c, h, w = input_tensor.shape
    align = 128
    PATCH_LONG = 1280

    pad_h = int(np.ceil(h / align) * align) - h
    pad_w = int(np.ceil(w / align) * align) - w
    img_padded = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect') if (pad_h > 0 or pad_w > 0) else input_tensor

    H_pad, W_pad = img_padded.shape[2], img_padded.shape[3]
    output_canvas = torch.zeros((b, 3, H_pad, W_pad), device=device)
    weight_canvas = torch.zeros((b, 1, H_pad, W_pad), device=device)

    is_h_long = h >= w
    if is_h_long:
        patch_h, patch_w = PATCH_LONG, W_pad
        y_list, x_list = [0, H_pad - patch_h], [0]
    else:
        patch_h, patch_w = H_pad, PATCH_LONG
        y_list, x_list = [0], [0, W_pad - patch_w]

    overlap_val = 128

    for ys in y_list:
        for xs in x_list:
            ye, xe = ys + patch_h, xs + patch_w
            patch_input = img_padded[:, :, ys:ye, xs:xe]

            fade_sides = (ys > 0, ye < H_pad, xs > 0, xe < W_pad)

            with autocast():
                patch_out = model(patch_input)
                if isinstance(patch_out, (list, tuple)):
                    patch_out = patch_out[0]
            patch_out = patch_out.float()

            mask = get_adaptive_mask(patch_h, patch_w, overlap_val, overlap_val, fade_sides, device)
            output_canvas[:, :, ys:ye, xs:xe] += patch_out * mask
            weight_canvas[:, :, ys:ye, xs:xe] += mask

    result = output_canvas / (weight_canvas + 1e-8)
    return result[:, :, :h, :w]

def do_inference(model, input_tensor, device, use_whole_image):
    if use_whole_image:
        try:
            return inference_whole(model, input_tensor)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"[GPU] OOM during whole-image inference, falling back to split inference...")
                torch.cuda.empty_cache()
                return inference_split2(model, input_tensor, device)
            else:
                raise
    else:
        return inference_split2(model, input_tensor, device)

def save_high_quality_jpg(tensor, save_path):
    tensor = torch.clamp(tensor, 0, 1)
    res_img = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    res_img_bgr = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
    params = [
        int(cv2.IMWRITE_JPEG_QUALITY), 100,
        int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,
        int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444
    ]
    cv2.imwrite(save_path, res_img_bgr, params)

def worker_process(gpu_id, scene_ids, return_dict, config, input_root, checkpoint_path, save_root, use_whole_image):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[Worker GPU {gpu_id}] Initializing model with default parameters...")
    model = AFUNet(**config).to(device)

    if not os.path.exists(checkpoint_path):
        print(f"[Worker GPU {gpu_id}] Error: Checkpoint not found at {checkpoint_path}")
        return_dict[gpu_id] = (0.0, 0)
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    # 彻底清理各种可能残留的前缀
    state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    try:
        # [极度关键] 必须为 False，才能吃下剔除了 attn_mask 的瘦身权重！
        model.load_state_dict(state_dict, strict=False)
        print(f"[Worker GPU {gpu_id}] Checkpoint loaded successfully.")
    except RuntimeError as e:
        print(f"[Worker GPU {gpu_id}] Failed to load checkpoint: {e}")
        return_dict[gpu_id] = (0.0, 0)
        return

    model.eval()
    worker_time = 0.0
    processed_count = 0

    with torch.no_grad():
        for scene_id in scene_ids:
            scene_path = os.path.join(input_root, scene_id)
            save_path = os.path.join(save_root, f"{scene_id}.jpg")

            if not os.path.exists(scene_path):
                print(f"[Worker GPU {gpu_id}] Scene not found: {scene_path}, skipping.")
                continue

            input_tensor = process_input_data(scene_path).to(device)

            torch.cuda.synchronize(device)
            start_time = time.time()

            output = do_inference(model, input_tensor, device, use_whole_image)

            torch.cuda.synchronize(device)
            inference_time = time.time() - start_time
            worker_time += inference_time

            save_high_quality_jpg(output, save_path)
            processed_count += 1

            print(f"[GPU {gpu_id}] Scene {scene_id} | Size: {list(input_tensor.shape[2:])} | Time: {inference_time:.4f}s")

    return_dict[gpu_id] = (worker_time, processed_count)

def main():
    args = parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    submission_zip_path = os.path.join(args.save_root, args.zip_name)

    print("==========================================")
    print(f"  Input Root    : {args.input_root}")
    print(f"  Checkpoint    : {args.checkpoint}")
    print(f"  Save Root     : {args.save_root}")
    print(f"  Input Frames  : {INPUT_INDICES}")
    print(f"  Whole Image   : {'ON (OOM fallback enabled)' if args.whole_image else 'OFF (split mode)'}")
    print(f"  GPUs          : {args.gpu_list}")
    print("==========================================")

    all_scenes = [f"{i:03d}" for i in range(1, 101)]
    num_gpus = len(args.gpu_list)
    chunk_size = ceil(len(all_scenes) / num_gpus)
    scene_chunks = [all_scenes[i:i + chunk_size] for i in range(0, len(all_scenes), chunk_size)]

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    start_total_time = time.time()

    for idx, gpu_id in enumerate(args.gpu_list):
        if idx < len(scene_chunks):
            p = mp.Process(
                target=worker_process,
                args=(gpu_id, scene_chunks[idx], return_dict, MODEL_CONFIG,
                      args.input_root, args.checkpoint, args.save_root, args.whole_image)
            )
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - start_total_time
    total_time = sum(v[0] for v in return_dict.values() if v)
    total_count = sum(v[1] for v in return_dict.values() if v)
    avg_time = total_time / total_count if total_count > 0 else 0.0

    print(f"\nAll GPUs finished. Wall-clock time: {elapsed:.4f}s")
    print(f"Average network runtime per image: {avg_time:.4f}s")

    # [修复] 补上要求的 Description 文本
    readme_path = os.path.join(args.save_root, "readme.txt")
    with open(readme_path, "w") as f:
        f.write(
            f"runtime per image [s] : {avg_time:.4f}\n"
            f"CPU[1] / GPU[0] : 0\n"
            f"Extra Data [1] / No Extra Data [0] : 0\n"
        )

    print(f"Zipping submission to {submission_zip_path}...")
    with zipfile.ZipFile(submission_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(readme_path, arcname="readme.txt")
        for scene_id in all_scenes:
            file_path = os.path.join(args.save_root, f"{scene_id}.jpg")
            if os.path.exists(file_path):
                zipf.write(file_path, arcname=f"{scene_id}.jpg")

    print("Done! Submission ready.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()