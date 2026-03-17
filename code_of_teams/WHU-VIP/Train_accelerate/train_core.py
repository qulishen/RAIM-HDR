# -*- coding: utf-8 -*-
"""
Training Entry Point — NTIRE 2026 RAIM Track2 (HDR Reconstruction)
Team: whu-vip

Description:
    Orchestrates the full training pipeline using HuggingFace Accelerate:
      - Multi-GPU DDP training with fixed fp16 mixed precision
      - Dynamic model loading from config
      - EMA (Exponential Moving Average) shadow model for stable weights
      - Cosine annealing LR scheduler
      - Checkpoint save / resume functionalities
      - torch.compile support for extreme speedup
"""

import os
import argparse
import yaml
import copy
import importlib
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from trainer import Trainer
from utils.datasets import RAIMTrainDataset, RepeatDataset
from utils.utils import get_logger_and_writer, save_checkpoint
from utils.metrics import NTIREScoreLoss


# =============================================================================
# Helpers
# =============================================================================

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model_class(filename, class_name):
    try:
        module = importlib.import_module(f"models.{filename}")
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"Cannot import models/{filename}.py: {e}")
    except AttributeError:
        raise AttributeError(f"Class '{class_name}' not found in models/{filename}.py")


def load_weights(model, checkpoint_path, logger=None):
    """Load weights strictly, stripping DDP / torch.compile prefixes."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    state_dict = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    if logger:
        logger.info(f"Loaded weights from: {checkpoint_path}")


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================

class ModelEMA:
    """
    Exponential Moving Average shadow model.
    Maintains a smoothed copy of the training model; does not participate in backprop.
    """
    def __init__(self, model, decay=0.999, device=None):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        if device is not None:
            self.module.to(device=device)
        for p in self.module.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        """EMA update: shadow = shadow * decay + model * (1 - decay)"""
        # Unwrap model in case it's compiled or DDP wrapped
        source_model = model.module if hasattr(model, 'module') else model
        
        for ema_v, model_v in zip(self.module.state_dict().values(), source_model.state_dict().values()):
            # Copy values directly (assumes they are on the same device via Accelerate)
            ema_v.copy_(ema_v * self.decay + (1.0 - self.decay) * model_v.to(ema_v.device))


# =============================================================================
# Main Training Routine
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NTIRE 2026 RAIM Track2 — Training")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    cli_args = parser.parse_args()

    cfg = load_config(cli_args.config)

    # --- Accelerate init ---
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision='fp16',
        log_with='tensorboard',
        project_dir=cfg['paths']['log_root'],
        kwargs_handlers=[ddp_kwargs]
    )

    local_rank = accelerator.process_index
    device = accelerator.device
    log_dir = os.path.join(cfg['paths']['log_root'], cfg['name'])

    dummy_args = argparse.Namespace()
    dummy_args.logdir = log_dir
    dummy_args.seed = cfg['seed']

    logger, writer = get_logger_and_writer(dummy_args, local_rank)
    if cfg['seed'] is not None:
        set_seed(cfg['seed'] + local_rank)

    if accelerator.is_main_process:
        logger.info(f"Config loaded from: {cli_args.config}")
        logger.info(f"Device: {device} | Mixed precision: fp16 (fixed)")

    # --- Model Initialization ---
    model_cfg = cfg['model'].copy()
    model_filename = model_cfg.pop('filename', model_cfg.get('type'))
    model_class_name = model_cfg.pop('type')
    ModelClass = load_model_class(model_filename, model_class_name)
    
    if 'img_size' not in model_cfg:
        model_cfg['img_size'] = cfg['data']['crop_size']
    model = ModelClass(**model_cfg)

    # --- EMA Initialization ---
    use_ema = cfg['train'].get('use_ema', False)
    ema_decay = cfg['train'].get('ema_decay', 0.999)
    ema_model = None
    if use_ema:
        ema_model = ModelEMA(model, decay=ema_decay, device=device)
        if accelerator.is_main_process:
            logger.info(f"EMA enabled (decay={ema_decay})")

    # --- Pretrained backbone (only when not resuming) ---
    resume_path = cfg['paths'].get('resume_path', None)
    pretrained_backbone = cfg['paths'].get('pretrained_backbone', None)

    if resume_path is None and pretrained_backbone and os.path.isfile(pretrained_backbone):
        if accelerator.is_main_process:
            logger.info("Loading pretrained backbone...")
        load_weights(model, pretrained_backbone, logger if accelerator.is_main_process else None)
        if use_ema and ema_model is not None:
            ema_model.update(model)

    # --- Optimizer & Loss ---
    use_lpips_weighted = cfg['train'].get('use_lpips_weighted', False)
    criterion_score = NTIREScoreLoss(device=device, lpips_weighted=use_lpips_weighted)
    base_lr = float(cfg['train']['lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-8)
    if accelerator.is_main_process:
        logger.info(f"Optimizer: Adam (lr={base_lr})")

    # --- Resume Logic ---
    start_epoch = 1
    global_step = None
    optimizer_state = None
    scheduler_state = None

    if resume_path and os.path.isfile(resume_path):
        if accelerator.is_main_process:
            logger.info(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        is_full_ckpt = isinstance(ckpt, dict) and 'state_dict' in ckpt

        sd = ckpt['state_dict'] if is_full_ckpt else ckpt
        sd = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)

        if use_ema and ema_model is not None:
            if is_full_ckpt and 'ema_state_dict' in ckpt:
                ema_sd = {k.replace('module.', '').replace('_orig_mod.', ''): v
                          for k, v in ckpt['ema_state_dict'].items()}
                ema_model.module.load_state_dict(ema_sd, strict=False)
                if accelerator.is_main_process:
                    logger.info("EMA state restored from checkpoint.")
            else:
                ema_model.module.load_state_dict(model.state_dict())
                if accelerator.is_main_process:
                    logger.info("No EMA state in checkpoint; initialized from model weights.")

        if is_full_ckpt and 'epoch' in ckpt:
            start_epoch = int(ckpt['epoch']) + 1
            global_step = int(ckpt.get('global_step', 0))
            optimizer_state = ckpt.get('optimizer', None)
            scheduler_state = ckpt.get('scheduler', None)

    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            if accelerator.is_main_process:
                logger.info("Optimizer state restored.")
        except Exception as e:
            if accelerator.is_main_process:
                logger.warning(f"Optimizer restore failed: {e}. Using fresh optimizer.")

    # --- Scheduler ---
    if start_epoch > 1:
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['train']['epochs'],
        eta_min=float(cfg['train']['min_lr']),
        last_epoch=start_epoch - 2
    )
    if scheduler_state is not None:
        try:
            scheduler.load_state_dict(scheduler_state)
        except Exception:
            pass

    # --- Dataset & DataLoader ---
    dataset_repeats = cfg['data'].get('dataset_repeats', 3)
    use_strong_aug = cfg['data'].get('use_strong_aug', False)
    raw_ds = RAIMTrainDataset(
        data_root=cfg['paths']['train_dir'],
        crop_size=cfg['data']['crop_size'],
        use_strong_aug=use_strong_aug
    )
    train_dataset = RepeatDataset(raw_ds, repeats=dataset_repeats) if dataset_repeats > 1 else raw_ds

    if accelerator.is_main_process:
        logger.info(f"Dataset: {cfg['paths']['train_dir']} | "
                    f"Base: {len(raw_ds)} | Repeats: {dataset_repeats} | Total: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=True,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        persistent_workers=(cfg['data']['num_workers'] > 0)
    )

    # --- [Fix 1] torch.compile MUST happen BEFORE accelerator.prepare ---
    if hasattr(torch, 'compile'):
        if accelerator.is_main_process:
            logger.info("=" * 60)
            logger.info("[EN] Compiling model with torch.compile...")
            logger.info("[EN] This JIT optimization process may take 3-5 minutes.")
            logger.info("[EN] It is completely normal for the program to appear 'stuck' here. Please be patient!")
            logger.info("-" * 60)
            logger.info("[CN] 正在使用 torch.compile 进行图编译优化...")
            logger.info("[CN] 该底层 C++ 内核生成过程通常需要 3 到 5 分钟。")
            logger.info("[CN] 程序在这里看起来像卡住了一样是【完全正常】的现象，请耐心等待！")
            logger.info("=" * 60)
            
        # Compile the raw model first to prevent graph breaks with DDP
        model = torch.compile(model)

    # --- [Fix 2] Accelerate prepare MUST include the scheduler ---
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    if global_step is None:
        global_step = max(0, (start_epoch - 1) * len(train_loader))

    # --- Trainer Initialization ---
    trainer = Trainer(
        cfg=cfg, accelerator=accelerator, model=model,
        train_loader=train_loader, optimizer=optimizer, scheduler=scheduler,
        criterion_score=criterion_score, logger=logger, writer=writer,
        ema_model=ema_model
    )
    trainer.global_step = global_step

    if accelerator.is_main_process:
        logger.info(f"Starting training — total epochs: {cfg['train']['epochs']}")

    # --- Training Loop ---
    for epoch in range(start_epoch, cfg['train']['epochs'] + 1):
        train_loss = trainer.train_epoch(epoch)
        trainer.scheduler.step()
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            loss_val = train_loss if isinstance(train_loss, (float, int)) else (
                train_loss.item() if hasattr(train_loss, 'item') else 'N/A')

            loss_log_path = os.path.join(log_dir, 'train_loss_log.txt')
            with open(loss_log_path, 'a') as f:
                f.write(f"Epoch {epoch}: Train Loss = {loss_val}\n")
            logger.info(f"Epoch {epoch} done. Loss: {loss_val}")

            # Safely unwrap the model for saving
            unwrapped = accelerator.unwrap_model(trainer.model)
            if hasattr(unwrapped, '_orig_mod'):
                unwrapped = unwrapped._orig_mod

            save_checkpoint(
                dummy_args, epoch, trainer.global_step,
                unwrapped, trainer.optimizer, trainer.scheduler,
                ema_model=trainer.ema_model
            )

        accelerator.wait_for_everyone()

    if accelerator.is_main_process and writer:
        writer.close()


if __name__ == '__main__':
    main()