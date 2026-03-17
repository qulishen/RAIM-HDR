# -*- coding: utf-8 -*-
"""NTIRE score-based composite loss (PSNR + SSIM + LPIPS). — Team: whu-vip"""

import torch
import torch.nn as nn
import pyiqa

class NTIREScoreLoss(nn.Module):
    """Differentiable approximation of the NTIRE 2026 RAIM evaluation score.

    Normalizes PSNR, SSIM, and LPIPS into [0, 1] and computes a weighted sum.
    The final loss is: loss = offset - (score / norm_factor)

    Two weighting modes:
      Standard (default) : PSNR=30,  SSIM=22.5, LPIPS=30   (max ~82.5)
      LPIPS-weighted     : PSNR=5,   SSIM=2.5,  LPIPS=75   (max ~82.5)
    """

    def __init__(self, device, lpips_weighted=False):
        super().__init__()
        self.device = device

        if lpips_weighted:
            self.w_psnr, self.w_ssim, self.w_lpips = 5.0, 2.5, 75.0
        else:
            self.w_psnr, self.w_ssim, self.w_lpips = 30.0, 22.5, 30.0

        self.psnr_range  = (0.0,  50.0)
        self.ssim_range  = (0.5,   1.0)
        self.lpips_range = (0.0,   1.0)
        self.norm_factor = 75.0
        self.offset      = 1.1

        # [修复] 强制将模型加载到当前子进程的特定 device 上，避免多卡 DDP 冲突死锁
        self.lpips_loss  = pyiqa.create_metric('lpips', device=self.device, as_loss=True).to(self.device)
        self.ssim_metric = pyiqa.create_metric('ssim',  device=self.device, as_loss=True).to(self.device)

        # 冻结所有评价指标的权重，防止反向传播时意外更新
        for param in self.lpips_loss.parameters():
            param.requires_grad = False
        for param in self.ssim_metric.parameters():
            param.requires_grad = False

    @staticmethod
    def _norm(score, lo, hi):
        return (score - lo) / (hi - lo)

    def forward(self, pred, target):
        """
        Returns:
            loss      (Tensor): scalar loss for backprop, range ~[0, 1.1]
            raw_score (Tensor): interpretable score, range ~[0, 82.5]
        """
        pred   = pred.float()
        target = target.float()

        # PSNR
        mse      = torch.clamp(torch.mean((pred - target) ** 2, dim=[1, 2, 3]), min=1e-8)
        psnr_val = -10.0 * torch.log10(mse)
        # 展平 PSNR 到标量
        psnr_val = psnr_val.mean()

        # SSIM & LPIPS
        # [修复] 确保所有返回的指标值都是展平的标量，防止 DDP 梯度聚合报错
        ssim_val  = self.ssim_metric(pred, target).mean()
        lpips_val = self.lpips_loss(pred, target).mean()

        # Normalize to [0, 1]
        s_psnr  = self._norm(psnr_val,                    *self.psnr_range)
        s_ssim  = self._norm(ssim_val,                    *self.ssim_range)
        s_lpips = self._norm(1.0 - lpips_val / 0.4,      *self.lpips_range)

        raw_score = self.w_psnr * s_psnr + self.w_ssim * s_ssim + self.w_lpips * s_lpips
        loss      = self.offset - raw_score / self.norm_factor
        
        return loss, raw_score