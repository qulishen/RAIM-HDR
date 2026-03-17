"""
We refer the code made from
https://github.com/z-bingo/kernel-prediction-networks-PyTorch/blob/master/train_eval_syn.py

eval2.py - 带重叠融合的分块推理版本，消除接缝问题
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import numpy as np
from fvcore.nn import FlopCountAnalysis, flop_count_table
import os, time
import math

from torchvision.transforms import transforms
to_pil_image = transforms.ToPILImage()
from DataLoader.custom_data_class import CustomDataset
# from models.unet_model import UNet
from models.Model_06_FRNet import flow_restormer as My_model

from utils.utils import *
from utils.checkpoint import *
from utils.brightness_align import gamma_linearize_tensor, gamma_encode_tensor

def tile_process_v2(model, input_tensor, tile_size, stride):
    """
    改进版：双向线性渐变权重融合 (无亮度对齐，纯加权合并)
    stride = tile_size // 2
    """
    b, t, c, h, w = input_tensor.shape
    device = input_tensor.device
    
    # 1. 初始化全图画布和权重图
    full_output = torch.zeros((b, 3, h, w)).to(device)
    full_weight = torch.zeros((b, 3, h, w)).to(device) # 使用 3 通道权重方便广播计算

    # 2. 生成 2D 权重矩阵 (余弦窗，比线性更平滑)
    # 边缘处权重为 0，中心为 1
    mask_h = torch.sin(torch.linspace(0, math.pi, tile_size)).to(device)
    mask_w = torch.sin(torch.linspace(0, math.pi, tile_size)).to(device)
    mask_2d = mask_h.reshape(tile_size, 1) * mask_w.reshape(1, tile_size)
    mask_2d = mask_2d.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

    # 3. 计算坐标列表 (考虑边界对齐)
    h_list = list(range(0, h - tile_size + 1, stride))
    if h_list[-1] != h - tile_size: h_list.append(h - tile_size)
    
    w_list = list(range(0, w - tile_size + 1, stride))
    if w_list[-1] != w - tile_size: w_list.append(w - tile_size)

    # 4. 扫描推理并加权累加
    with torch.no_grad():
        for y in h_list:
            for x in w_list:
                # 裁剪输入分块
                tile_in = input_tensor[:, :, :, y:y+tile_size, x:x+tile_size]
                
                # 模型推理
                tile_out = model(tile_in) # 输出通常是 [1, 3, tile_size, tile_size]
                
                # 加权累加到全图
                full_output[:, :, y:y+tile_size, x:x+tile_size] += tile_out * mask_2d
                full_weight[:, :, y:y+tile_size, x:x+tile_size] += mask_2d

    # 5. 除以权重得到平均值 (避免除以 0)
    output = full_output / (full_weight + 1e-8)
    
    return torch.clamp(output, 0.0, 1.0)


    """
    实现细化后的 TLC 推理逻辑
    """
    if stride==0:
        stride = tile_size // 2
    b, t, c, h, w = input_tensor.shape
    device = input_tensor.device
    
    # 计算分块坐标
    h_list = list(range(0, h - tile_size + 1, stride))
    if h_list[-1] != h - tile_size: h_list.append(h - tile_size)
    w_list = list(range(0, w - tile_size + 1, stride))
    if w_list[-1] != w - tile_size: w_list.append(w - tile_size)
    
    rows, cols = len(h_list), len(w_list)
    tiles = [[None for _ in range(cols)] for _ in range(rows)]
    
    # 1. 逐块推理并存储
    with torch.no_grad():
        for r, y in enumerate(h_list):
            for c_idx, x in enumerate(w_list):
                tile_in = input_tensor[:, :, :, y:y+tile_size, x:x+tile_size]
                tile_out = model(tile_in) # [1, 3, H, W]
                tiles[r][c_idx] = tile_out
                
    # 2. 计算相邻块亮度差
    diffs_x = [[0.0 for _ in range(cols-1)] for _ in range(rows)]
    diffs_y = [0.0 for _ in range(rows-1)]
    
    for r in range(rows):
        for c_idx in range(cols-1):
            diffs_x[r][c_idx] = compute_bright_difference(tiles[r][c_idx], tiles[r][c_idx+1], 'horizontal', stride)
            
    for r in range(rows-1):
        diffs_y[r] = compute_bright_difference(tiles[r][0], tiles[r+1][0], 'vertical', stride)
        
    # 3. 亮度对齐
    tiles = align_brightness_grid(tiles, diffs_x, diffs_y)
    
    # 4. 最终 TLC 融合
    full_output = torch.zeros((b, 3, h, w)).to(device)
    full_weight = torch.zeros((b, 1, h, w)).to(device)
    
    # 生成软权重矩阵
    mask = torch.ones((1, 1, tile_size, tile_size)).to(device)
    for i in range(tile_size): # 简单的线性渐变权重
        v = min(i, tile_size - 1 - i) / (tile_size / 2)
        mask[:, :, i, :] *= v
        mask[:, :, :, i] *= v

    for r, y in enumerate(h_list):
        for c_idx, x in enumerate(w_list):
            full_output[:, :, y:y+tile_size, x:x+tile_size] += tiles[r][c_idx] * mask
            full_weight[:, :, y:y+tile_size, x:x+tile_size] += mask
            
    return full_output / (full_weight + 1e-8)

import torch
import math

def tile_process_v3(model, input_tensor, tile_size, overlap):
    """
    v3 改进版：支持自定义 overlap 的分块推理
    - tile_size: 分块大小 (如 768)
    - overlap: 重叠像素 (如 64 或 128)
    """
    b, t, c, h, w = input_tensor.shape
    device = input_tensor.device
    
    # 1. 计算步长 (Stride)
    stride = tile_size - overlap
    
    # 2. 初始化全图画布和权重图
    # 结果通常为 RGB (3通道)
    full_output = torch.zeros((b, 3, h, w), device=device)
    full_weight = torch.zeros((b, 3, h, w), device=device)

    # 3. 生成平滑权重矩阵 (余弦窗)
    # 为了防止边缘处权重完全为 0 导致数值空洞，我们给 linspace 加一个微小的 offset
    # 或者使用更稳健的计算方式
    mask_h = torch.sin(torch.linspace(0.01, math.pi - 0.01, tile_size)).to(device)
    mask_w = torch.sin(torch.linspace(0.01, math.pi - 0.01, tile_size)).to(device)
    mask_2d = (mask_h.reshape(tile_size, 1) * mask_w.reshape(1, tile_size))
    
    # 提高余弦窗的阶数，使中心部分权重更集中，边缘更平滑
    mask_2d = mask_2d.pow(2.0).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

    # 4. 计算坐标列表 (确保覆盖全图，且最后一块对齐右下角)
    def get_coords(full_size, tile_size, stride):
        coords = []
        pos = 0
        while pos + tile_size <= full_size:
            coords.append(pos)
            pos += stride
        if coords[-1] != full_size - tile_size:
            coords.append(full_size - tile_size)
        return coords

    h_list = get_coords(h, tile_size, stride)
    w_list = get_coords(w, tile_size, stride)

    # 5. 扫描推理并加权累加
    model.eval()
    with torch.no_grad():
        for y in h_list:
            for x in w_list:
                # 裁剪输入分块 [B, T, C, tile_H, tile_W]
                tile_in = input_tensor[:, :, :, y:y+tile_size, x:x+tile_size]
                
                # 模型推理 (确保输出格式与输入空间尺寸一致)
                # 如果你的 Restormer 包含混合精度，请在这里嵌套 autocast
                tile_out = model(tile_in) 
                
                # 加权累加到全图
                # tile_out 形状应为 [B, 3, tile_H, tile_W]
                full_output[:, :, y:y+tile_size, x:x+tile_size] += tile_out * mask_2d
                full_weight[:, :, y:y+tile_size, x:x+tile_size] += mask_2d

    # 6. 归一化并清理显存
    output = full_output / (full_weight + 1e-8)
    
    # 显存回收
    del full_output, full_weight
    
    return torch.clamp(output, 0.0, 1.0)

def get_tile_size1(model, full_h, full_w, device, ceil_size=16):
    """
    (1) 自动探测不产生 OOM 的分块尺寸
    """
    h_, w_ = full_h, full_w
    print(f"Starting tile size detection from {h_}x{w_}...")
    
    while True:
        h_ = math.ceil(h_ / ceil_size) * ceil_size
        w_ = math.ceil(w_ / ceil_size) * ceil_size
        
        try:
            test_input = torch.ones(1, 7, 3, h_, w_).to(device)
            with torch.no_grad():
                _ = model(test_input)
            torch.cuda.empty_cache()
            print(f"Safe tile size detected: {h_}x{w_}")
            return h_, w_
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                h_, w_ = h_ // 2, w_ // 2 # 缩小一半
                if h_ < ceil_size * 4: # 设置最小阈值防止死循环
                    raise RuntimeError("Tile size too small, even 64x64 OOM.")
                print(f"OOM at previous size, trying: {h_}x{w_}")
            else:
                raise e

def get_tile_size(model, full_h, full_w, device, ceil_size=16):
    """
    (1) 自动探测不产生 OOM 的分块尺寸
    按照梯度递减: 1024 -> 768 -> 512 -> 256 -> 128 -> 64
    """
    # 定义搜索梯度
    candidate_sizes = [1024, 768, 512, 256, 128, 64]
    
    # 首先根据图片的实际尺寸过滤掉比原图大太多的候选尺寸（可选）
    # 但通常为了简单，我们直接从列表里最大的开始试
    
    print(f"Starting tile size detection. Image size: {full_h}x{full_w}")
    
    for size in candidate_sizes:
        # 确保 size 是 ceil_size 的倍数
        cur_size = math.ceil(size / ceil_size) * ceil_size
        
        # 如果当前探测尺寸已经大于原图且原图能跑通，其实可以直接用原图
        # 但为了稳定，我们统一测试这个 cur_size
        
        try:
            # 这里的 7 是输入通道数（根据你的模型输入调整，如 3 或 7）
            test_input = torch.ones(1, 7, 3, cur_size, cur_size).to(device)
            with torch.no_grad():
                _ = model(test_input)
            
            # 清理测试占用的显存
            del test_input
            torch.cuda.empty_cache()
            
            # 最终确定的尺寸不应超过原图实际大小
            final_h = min(cur_size, math.ceil(full_h / ceil_size) * ceil_size)
            final_w = min(cur_size, math.ceil(full_w / ceil_size) * ceil_size)
            
            print(f"Safe tile size detected: {final_h}x{final_w} (derived from buffer {cur_size})")
            return final_h, final_w
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                print(f"OOM at {cur_size}x{cur_size}, trying next smaller size...")
                continue
            else:
                raise e
                
    raise RuntimeError("Even the minimum tile size (64x64) causes OOM. Please check GPU memory.")

# 保留原来的无重叠版本供参考
def tiled_inference(model, input_tensor, tile_h, tile_w):
    """
    (2) & (3) 正式分块推理并直接覆盖拼接（无重叠版本）
    """
    b, t, c, h, w = input_tensor.shape
    output = torch.zeros((b, 3, h, w)).to(input_tensor.device)
    
    # 计算步长（无重叠扫描，最后一行/列会自动修正）
    for y in range(0, h, tile_h):
        for x in range(0, w, tile_w):
            # 修正越界：如果超出原图范围，向左/向上移动起始点
            y_start = y
            y_end = y_start + tile_h
            if y_end > h:
                y_end = h
                y_start = max(0, y_end - tile_h)
                
            x_start = x
            x_end = x_start + tile_w
            if x_end > w:
                x_end = w
                x_start = max(0, x_end - tile_w)
            
            # 提取分块数据
            tile = input_tensor[:, :, :, y_start:y_end, x_start:x_end]
            
            # 推理并将结果直接覆盖到对应位置
            with torch.no_grad():
                pred_tile = model(tile)
                output[:, :, y_start:y_end, x_start:x_end] = pred_tile
                
    return output


def eval(args, cuda, input_gt=True, mGPU=True):
    # ==================== 修改路径 ====================
    checkpoint_dir = args.ckpt_dir                              # 权重目录
    eval_dir = args.eval_dir                                    # 结果保存目录
    test_root = args.test_root                                  # 测试数据路径
    checkpoint_path = os.path.join(checkpoint_dir, args.name)   # 具体权重文件
    # ==================== 修改路径 ====================
    
    if not os.path.exists(checkpoint_dir) or len(os.listdir(checkpoint_dir)) == 0:
        print('There is no any checkpoint file in path:{}'.format(checkpoint_dir))
    
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    # dataset and dataloader
    data_set = CustomDataset(root_dir=test_root, transform=transforms.ToTensor(), train=False)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False)
    print("Length of the data_loader :", len(data_loader))

    """
        Your model will be loaded here, via submitted pytorch code and trained parameters.
        You may upload zip file containing my_network.py and my_parameters.pth.tar to the server. 
        The specific guideline for how to submit your model will be provided later. 
    """
    ## your model here. ####
    model = My_model(in_chans= 64, 
        embed_dim= 60, 
        dim= 48, 
        num_blocks=[2, 2, 2, 2], 
        num_refinement_blocks= 2, 
        heads=[1, 2, 4, 8], 
        ffn_expansion_factor = 2.66, 
        bias=False,LayerNorm_type='BiasFree')

    # load trained model parameters
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到指定的权重文件: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    global_step = checkpoint['global_iter']
    best_loss = checkpoint['best_loss']
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model = model.cuda()

    if mGPU:
        model = nn.DataParallel(model)

    print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
    print('The model has been completely loaded from the user submission.')

    # parameters and flops
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full_h, full_w = 2040, 2040
    safe_h, safe_w = get_tile_size(model, full_h, full_w, device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total # of model parameters : {num_params / 1000 / 1000 :.3f}(M)")
    print('\n-------Evaluation started -------\n')

    # switch the eval mode
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        psnr = 0.0
        ssim = 0.0

        for i, (burst_noise, gt) in enumerate(data_loader):

            t0 = time.time()
            if cuda:
                burst_noise = burst_noise.cuda()
                gt = gt.cuda()

            burst_noise = burst_noise.squeeze(2)
            
            # 使用带重叠的分块推理
            assert safe_h == safe_w
            pred = tile_process_v3(model, burst_noise, tile_size=safe_h, overlap=safe_h // args.overlap_num1 * args.overlap_num2)
            pred = torch.clamp(pred, 0.0, 1.0)
            
            if input_gt:
                psnr_t = calculate_psnr(pred.unsqueeze(1), gt.unsqueeze(1))
                ssim_t = calculate_ssim(pred.unsqueeze(1), gt.unsqueeze(1))
                psnr += psnr_t
                ssim += ssim_t
            pred = torch.clamp(pred, 0.0, 1.0)
            t1 = time.time()

            if cuda:
                pred = pred.cpu()
                gt = gt.cpu()
                burst_noise = burst_noise.cpu()
            if input_gt:
                print('{}-th image completed.\t| PSNR: {:.2f}dB\t| SSIM: {:.4f}\t| time: {:.2f}s'.format(i, psnr_t, ssim_t, t1 - t0))

            # 保存结果
            # 获取场景名
            scene_folders = sorted([d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))])
            if i < len(scene_folders):
                current_scene = scene_folders[i]
            else:
                current_scene = f"{i:04d}"
            
            # 保存模型输出
            out_img = (pred[0] * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            save_file_path = os.path.join(eval_dir, f"{current_scene}.jpg")
            cv2.imwrite(save_file_path, out_img)
            # os.path.basename(eval_dir)
            sub_eval_dir = os.path.join(eval_dir, os.path.basename(eval_dir))
            os.makedirs(sub_eval_dir, exist_ok=True)
            sub_save_file_path = os.path.join(sub_eval_dir, f"{current_scene}.jpg")
            cv2.imwrite(sub_save_file_path, out_img)

            print(f'Scene {current_scene} saved to {save_file_path}')

    end_time = time.time()
    print('All images are OK, average PSNR: {:.2f}dB, SSIM: {:.4f}'.format(psnr/(i+1), ssim/(i+1)))
    print(f'Total Validation time : {end_time - start_time : .2f} seconds.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="模型推理脚本")

    ########### --- file paths --- ##########
    parser.add_argument("--ckpt_dir", type=str, default="./model_zoo",
                                help="后权重保存路径")
    parser.add_argument("--eval_dir", type=str, default="./output_eval",
                                help="测试集结果保存路径")
    parser.add_argument("--test_root", type=str, default="./dataset/testdata_phase3",
                                help="日志文件保存路径")
    parser.add_argument("--name", type=str, default="model_best.pth.tar",
                                help="测试后权重名称")

    ########### --- hyperparameters of evaluation --- ##########
    parser.add_argument("--overlap_num1", type=int, default=2)
    parser.add_argument("--overlap_num2", type=int, default=1)
    args = parser.parse_args()

    eval(args, cuda=True, input_gt=False, mGPU=1)

