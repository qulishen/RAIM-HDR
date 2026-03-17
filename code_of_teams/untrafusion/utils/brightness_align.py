"""
亮度对齐模块 - 从 alig/utils/image_utils.py 复制
基于整图平均亮度的曝光匹配，将所有帧对齐到最亮帧
"""
import torch
from torch import Tensor
from typing import List


def gamma_linearize_tensor(img: Tensor, gamma: float = 2.2) -> Tensor:
    """
    GPU Gamma 线性化：将 sRGB Tensor 转为线性空间
    
    Args:
        img: 输入 Tensor，值范围 [0, 1]，格式 (C, H, W) 或 (B, C, H, W)
        gamma: gamma 值，默认 2.2
        
    Returns:
        Tensor: 线性空间 Tensor，值范围 [0, 1]
    """
    img = torch.clamp(img, 0.0, 1.0)
    return torch.pow(img + 1e-8, gamma)


def gamma_encode_tensor(img: Tensor, gamma: float = 2.2) -> Tensor:
    """
    GPU Gamma 编码：将线性 Tensor 转为 sRGB
    
    Args:
        img: 线性空间 Tensor，值范围 [0, 1]
        gamma: gamma 值，默认 2.2
        
    Returns:
        Tensor: sRGB Tensor，值范围 [0, 1]
    """
    img = torch.clamp(img, 0.0, 1.0)
    return torch.pow(img + 1e-8, 1.0 / gamma)


def estimate_exposure_ratio_tensor(src_linear: Tensor, 
                                  target_linear: Tensor) -> float:
    """
    GPU 估计曝光比率：使用整图平均亮度
    
    Args:
        src_linear: 源线性 Tensor (C, H, W)
        target_linear: 目标线性 Tensor (C, H, W)
        
    Returns:
        float: 曝光比率
    """
    # 整图平均亮度
    src_gray = torch.mean(src_linear).item()
    target_gray = torch.mean(target_linear).item()
    
    if src_gray < 1e-6:
        return 1.0
    
    return target_gray / src_gray


def align_brightness_tensor(src: Tensor, 
                            target: Tensor,
                            gamma: float = 2.2) -> Tensor:
    """
    GPU 亮度对齐：将源图像曝光调整到与目标图像一致
    
    Args:
        src: 源 Tensor，值范围 [0, 1]，格式 (C, H, W)
        target: 目标 Tensor，值范围 [0, 1]，格式 (C, H, W)
        gamma: gamma 值
        
    Returns:
        Tensor: 对齐后的 sRGB Tensor，值范围 [0, 1]
    """
    # Gamma 线性化
    src_linear = gamma_linearize_tensor(src, gamma)
    target_linear = gamma_linearize_tensor(target, gamma)
    
    # 估计曝光比率 (整图平均)
    ratio = estimate_exposure_ratio_tensor(src_linear, target_linear)
    
    # 应用曝光调整
    aligned_linear = src_linear * ratio
    
    # 裁剪到合法范围
    aligned_linear = torch.clamp(aligned_linear, 0.0, 1.0)
    
    # Gamma 编码转回 sRGB
    aligned_srgb = gamma_encode_tensor(aligned_linear, gamma)
    
    return aligned_srgb


def align_to_brightest_tensor(images: List[Tensor],
                              gamma: float = 2.2) -> List[Tensor]:
    """
    GPU 将所有图像对齐到最亮的图像
    
    Args:
        images: Tensor 列表 [(C, H, W), ...]，值范围 [0, 1]
        gamma: gamma 值
        
    Returns:
        list: 对齐后的 Tensor 列表
    """
    if len(images) == 0:
        return []
    
    # 找到最亮的图像（亮度最高）
    brightest_idx = 0
    max_brightness = 0.0
    for i, img in enumerate(images):
        brightness = torch.mean(img).item()
        if brightness > max_brightness:
            max_brightness = brightness
            brightest_idx = i
    
    target = images[brightest_idx]
    
    # 对齐所有图像到最亮的
    aligned_images = []
    for i, img in enumerate(images):
        if i == brightest_idx:
            aligned_images.append(img)
        else:
            aligned_images.append(align_brightness_tensor(img, target, gamma))
    
    return aligned_images