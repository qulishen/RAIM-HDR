# -*- coding: utf-8 -*-
"""Utility functions: logging, TensorBoard writer, checkpoint save/load. — Team: whu-vip"""

import os
import logging
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter


class AverageMeter:
    """Tracks and computes running average of a scalar value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def get_logger_and_writer(args, local_rank):
    """Create a logger and TensorBoard SummaryWriter (main process only)."""
    logger = logging.getLogger(f"rank{local_rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    writer = None

    if local_rank == 0:
        os.makedirs(args.logdir, exist_ok=True)
        fh        = logging.FileHandler(os.path.join(args.logdir, 'train_system.log'), mode='a')
        ch        = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(fh)
            logger.addHandler(ch)
        writer = SummaryWriter(log_dir=os.path.join(args.logdir, 'tb_logs'))

    return logger, writer


def _slim_state_dict(state_dict):
    """Strip DDP/compile prefixes and remove attention mask buffers.

    attn_mask buffers are dynamically recomputed at runtime and only bloat
    the checkpoint file when saved; they should never be persisted.
    """
    clean = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '').replace('_orig_mod.', '')
        if 'attn_mask' in new_k:
            continue
        clean[new_k] = v
    return clean


def save_checkpoint(args, epoch, global_step, model, optimizer, scheduler, ema_model=None):
    """Save training checkpoint and slimmed inference-ready weight files.

    Always saves:
      - latest_checkpoint.pth   full state for resume (optimizer, scheduler, etc.)
      - latest_model.pth        slimmed model weights (backbone / inference use)
      - latest_ema_model.pth    slimmed EMA weights (if available)
    """
    # Safety unwrap DDP model
    if hasattr(model, 'module'):
        raw_state = model.module.state_dict()
    else:
        raw_state = model.state_dict()

    state = {
        'epoch':       epoch,
        'global_step': global_step,
        'state_dict':  raw_state,
        'optimizer':   optimizer.state_dict(),
        'args':        args,
    }
    
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
        
    if ema_model is not None:
        if hasattr(ema_model.module, 'module'):
            state['ema_state_dict'] = ema_model.module.module.state_dict()
        else:
            state['ema_state_dict'] = ema_model.module.state_dict()

    # Save the comprehensive checkpoint for resuming training
    torch.save(state, os.path.join(args.logdir, 'latest_checkpoint.pth'))

    # Save the strictly slimmed weights for clean inference testing
    torch.save(_slim_state_dict(raw_state),
               os.path.join(args.logdir, 'latest_model.pth'))

    # If EMA is enabled, save the slimmed EMA model directly as the final deliverable
    if ema_model is not None:
        if hasattr(ema_model.module, 'module'):
            ema_state = ema_model.module.module.state_dict()
        else:
            ema_state = ema_model.module.state_dict()
            
        torch.save(_slim_state_dict(ema_state),
                   os.path.join(args.logdir, 'latest_ema_model.pth'))


def move_optimizer_state_to_device(optimizer, device):
    """Move all optimizer tensor states to the target device."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)