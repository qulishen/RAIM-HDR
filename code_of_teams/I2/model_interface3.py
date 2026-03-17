import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import save_image
from models import *
import pyiqa
import argparse
torch.serialization.add_safe_globals([argparse.Namespace])

class DehazeModelInterface(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        # 1. 动态加载模型
        # ⚠️ 确保模型第一层 in_channels=15
        model_class_name = args.model.replace('-', '_')
        fusion_net = eval(model_class_name)()
        self.net = WaveletHDR2(fusion_net)

        # 2. 定义损失函数
        self.criterion = nn.L1Loss()
        # 基础loss
        self.l1_loss = nn.L1Loss()
        
        # SSIM loss
        self.ssim_loss = pyiqa.create_metric('ssim', as_loss=True, device=self.device)
        
        # LPIPS loss
        self.lpips_loss = pyiqa.create_metric('lpips', as_loss=True, device=self.device)
        
        # wavelet transform
        from pytorch_wavelets import DWTForward
        self.dwt = DWTForward(J=1, wave='haar')


        # 3. 初始化评价指标 (全部使用 pyiqa)
        # 评测阶段不需要反向传播，设置 as_loss=False 确保返回标准值并节省显存
        self.psnr_metric = pyiqa.create_metric('psnr', as_loss=True,device=self.device)
        self.ssim_metric = pyiqa.create_metric('ssim', as_loss=True,device=self.device)
        # LPIPS 默认使用 alexnet，pyiqa 会自动处理输入范围
        self.lpips_metric = pyiqa.create_metric('lpips', as_loss=True,device=self.device)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        source_img = batch['source']
        target_img = batch['target']

        output = self(source_img)
#        loss = self.criterion(output, target_img)
        # L1
        l1 = self.l1_loss(output, target_img)
        
        # convert to [0,1]
        out_01 = torch.clamp(output * 0.5 + 0.5, 0.0, 1.0)
        tgt_01 = torch.clamp(target_img * 0.5 + 0.5, 0.0, 1.0)
        
        # SSIM
        ssim = self.ssim_loss(out_01, tgt_01)
        
        # LPIPS
        lpips = self.lpips_loss(out_01, tgt_01)
        
        # Wavelet
        wavelet = self.wavelet_loss(output, target_img)
        
        loss = (
            l1
            + 0.1 * (1 - ssim)
            + 0.05 * lpips
            + 0.1 * wavelet
        )


#        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log_dict({
            "train_loss": loss,
            "train_l1": l1,
            "train_ssim": ssim,
            "train_lpips": lpips,
            "train_wavelet": wavelet
        }, on_step=True, prog_bar=True, logger=True)

        return loss

    # ---------------------------------------------------
    # 核心修改：使用 pyiqa 统一计算所有指标与比赛 SCORE
    # ---------------------------------------------------
    def _compute_metrics(self, output, target):
        # 1. 预处理：将 [-1, 1] 映射回 [0, 1]，并使用 clamp 防止越界
        # pyiqa 默认期望输入在 [0, 1] 范围内
        out_01 = torch.clamp(output * 0.5 + 0.5, 0.0, 1.0)
        tgt_01 = torch.clamp(target * 0.5 + 0.5, 0.0, 1.0)

        # 2. 计算基础指标 (pyiqa 内部会自动处理 device)
        psnr = self.psnr_metric(out_01, tgt_01)
        ssim = self.ssim_metric(out_01, tgt_01)
        lpips = self.lpips_metric(out_01, tgt_01)

        # 3. 计算比赛综合得分 SCORE
        # SCORE = 30*PSNR/50 + 22.5*(SSIM-0.5)/0.5 + 30*(1-LPIPS/0.4)
        score = 30.0 * (psnr / 50.0) + \
                22.5 * ((ssim - 0.5) / 0.5) + \
                30.0 * (1.0 - (lpips / 0.4))

        return score, psnr, ssim, lpips
    def wavelet_loss(self, pred, target):
    
        with torch.cuda.amp.autocast(enabled=False):
    
            pred = pred.float()
            target = target.float().detach()

    
            yl_p, yh_p = self.dwt(pred)
            yl_t, yh_t = self.dwt(target)
    
            loss = F.l1_loss(yl_p, yl_t)
    
            loss += F.l1_loss(yh_p[0], yh_t[0])
    
        return loss


    # ---------------------------------------------------
    # Validation Step
    # ---------------------------------------------------
    def validation_step(self, batch, batch_idx):
        source_img = batch['source']
        target_img = batch['target']

        # 推理并限制在 [-1, 1]
        output = self(source_img).clamp_(-1, 1)

        # 计算所有指标
        score, psnr, ssim, lpips = self._compute_metrics(output, target_img)

        # 记录日志 (sync_dist=True 用于多卡同步平均值)
        self.log_dict({
            'valid_score': score,
            'valid_psnr': psnr,
            'valid_ssim': ssim,
            'valid_lpips': lpips
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return score

    # ---------------------------------------------------
    # Test Step (包含保存图片)
    # ---------------------------------------------------
    def test_step(self, batch, batch_idx):
        source_img = batch['source']
        target_img = batch['target']

        # 获取文件名
        filename = batch.get('sample_name', [f'sample_{batch_idx}_{i}' for i in range(source_img.size(0))])

        # 推理
        output = self(source_img).clamp_(-1, 1)

        # 计算所有指标
        score, psnr, ssim, lpips = self._compute_metrics(output, target_img)

        # 记录测试日志
        self.log_dict({
            'test_score': score,
            'test_psnr': psnr,
            'test_ssim': ssim,
            'test_lpips': lpips
        }, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        # 保存图片逻辑
        save_dir = os.path.join('results', self.args.exp, self.args.model)
        os.makedirs(save_dir, exist_ok=True)

        for i in range(output.size(0)):
            img_tensor = output[i] * 0.5 + 0.5  # [-1, 1] -> [0, 1]
            name = filename[i] if isinstance(filename, list) else filename
            if not (name.endswith('.png') or name.endswith('.jpg')):
                name = f"{name}.jpg"

            save_path = os.path.join(save_dir, name)
            save_image(img_tensor, save_path)

        return score

    def configure_optimizers(self):
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.args.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.max_steps,
            eta_min=self.args.lr * 1e-2
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
