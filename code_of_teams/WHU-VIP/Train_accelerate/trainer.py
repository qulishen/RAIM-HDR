# -*- coding: utf-8 -*-
"""
Trainer — NTIRE 2026 RAIM Track2 (HDR Reconstruction)
Team: whu-vip

Description:
    Implements the per-epoch training loop with:
      - AMP (fp16) via HuggingFace Accelerate
      - Optional STE (Straight-Through Estimator) quantization-aware training
      - Optional EMA (Exponential Moving Average) model update
      - Gradient accumulation and gradient clipping
      - TensorBoard logging
"""

import time
import torch
from accelerate import Accelerator
from utils.utils import AverageMeter


def quantize_ste(x):
    """
    Simulate 8-bit JPEG quantization via Straight-Through Estimator (STE).
    Forward pass introduces rounding error; gradients pass through unmodified.
    """
    x = torch.clamp(x, 0.0, 1.0)
    x_quant = torch.round(x * 255.0) / 255.0
    return x + (x_quant - x).detach()


class Trainer:
    def __init__(self, cfg, accelerator, model, train_loader, optimizer, scheduler,
                 criterion_score, logger, writer, ema_model=None):
        self.cfg             = cfg
        self.accelerator     = accelerator
        self.model           = model
        self.ema_model       = ema_model
        self.train_loader    = train_loader
        self.optimizer       = optimizer
        self.scheduler       = scheduler
        self.criterion_score = criterion_score
        self.logger          = logger
        self.writer          = writer
        self.device          = accelerator.device
        self.local_rank      = accelerator.process_index
        self.global_step     = 0

        self.grad_accum_steps = self.cfg['train']['grad_accum_steps']
        self.log_interval     = self.cfg['train']['log_interval']
        self.use_ste_quant    = self.cfg['train'].get('use_ste_quant', False)

        # [修复] 动态将配置中的累加步数注入给 Accelerator，激活其自动梯度累加机制
        self.accelerator.gradient_accumulation_steps = self.grad_accum_steps

        if self.accelerator.is_main_process:
            self.logger.info(f"Trainer initialized. Mixed precision: {self.accelerator.mixed_precision}")
            self.logger.info(f"Gradient accumulation steps: {self.grad_accum_steps}")
            if self.use_ste_quant:
                self.logger.info("[*] STE Quantization-Aware Training is ENABLED.")

    def train_epoch(self, epoch):
        self.model.train()

        # 确保 DDP 模式下 DataLoader 每轮能打乱数据
        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        batch_time   = AverageMeter()
        losses_optim = AverageMeter()
        scores_orig  = AverageMeter()
        end = time.time()

        for batch_idx, batch_data in enumerate(self.train_loader):
            # Accelerate 会在这个上下文中自动处理梯度累加的 forward、backward 缩放和同步
            with self.accelerator.accumulate(self.model):
                inputs, targets = batch_data
                # [继承上一轮的修复] 将 uint8 归一化回 [0, 1] 区间
                inputs  = inputs.to(self.device, non_blocking=True).float() / 255.0
                targets = targets.to(self.device, non_blocking=True).float() / 255.0

                with self.accelerator.autocast():
                    pred = self.model(inputs)
                    pred_eval = quantize_ste(pred) if self.use_ste_quant else pred
                    loss_val, raw_score = self.criterion_score(pred_eval, targets)
                    
                    # [修复] 移除手动的 loss_val / grad_accum_steps，Accelerate 会自动接管缩放

                # 直接传入未缩放的 loss
                self.accelerator.backward(loss_val)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if self.accelerator.sync_gradients:
                    # Update EMA model after each true parameter update
                    if self.ema_model is not None:
                        base_model = self.accelerator.unwrap_model(self.model)
                        if hasattr(base_model, '_orig_mod'):
                            base_model = base_model._orig_mod
                        self.ema_model.update(base_model)

                    self.global_step += 1

            # [修复] 将日志更新移出 accumulate 上下文，以确保每个 batch 的 loss 和耗时都能被精准追踪
            losses_optim.update(loss_val.item(), inputs.size(0))
            scores_orig.update(-raw_score.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            # 仅在实际参数更新的那一步，并且到达 log_interval 时打印日志
            if self.accelerator.sync_gradients and self.accelerator.is_main_process and self.global_step % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                msg = (f'Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}]  '
                       f'Step: {self.global_step}  '
                       f'Time: {batch_time.val:.3f}  '
                       f'LR: {current_lr:.2e}  '
                       f'Loss: {losses_optim.val:.4f} ({losses_optim.avg:.4f})  '
                       f'Score: {scores_orig.val:.2f} ({scores_orig.avg:.2f})')
                self.logger.info(msg)

                if self.writer:
                    self.writer.add_scalar('Train/Loss',  losses_optim.val, self.global_step)
                    self.writer.add_scalar('Train/LR',    current_lr,       self.global_step)
                    self.writer.add_scalar('Train/Score', scores_orig.val,  self.global_step)

        return losses_optim.avg