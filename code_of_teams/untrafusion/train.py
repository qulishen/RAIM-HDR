"""
We refer the code made from  
https://github.com/z-bingo/kernel-prediction-networks-PyTorch/blob/master/train_eval_syn.py
"""

import torch.nn.functional as F

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import argparse

import os, sys, time, shutil
from accelerate import Accelerator

from PIL import Image
from torchvision.transforms import transforms
to_pil_image = transforms.ToPILImage()

from DataLoader.custom_data_class import CustomDataset
from models.Model_06_FRNet import flow_restormer as My_model
import pdb

from utils.utils import *
from utils.checkpoint import *
from piq import LPIPS
lpips_metric = LPIPS(replace_pooling=True, reduction='mean')

# NCCL 稳定性配置
os.environ['NCCL_TIMEOUT'] = '180'  # 30分钟超时
os.environ['NCCL_IB_DISABLE'] = '0'
os.environ['NCCL_NET_GDR_LEVEL'] = '2'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步执行，便于调试

def custom_loss(pred, gt):
    """
    基于比赛指标的损失函数
    """
    # MSE Loss (PSNR 的一部分)
    mse = F.mse_loss(pred, gt)
    
    # SSIM Loss
    ssim_val = calculate_ssim(pred.unsqueeze(1), gt.unsqueeze(1))
    ssim_loss = 1 - ssim_val
    
    # LPIPS Loss
    lpips_val = lpips_metric(pred, gt)
    lpips_loss = lpips_val / 0.4
    
    # 组合（按比赛权重）
    loss = (30/50) * mse + 22.5 * ssim_loss + 30 * lpips_loss
    
    return loss, ssim_val, lpips_val

def train(args, num_threads, cuda, restart_train, mGPU):
    torch.set_num_threads(num_threads)

    # initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision = args.mixed_precision
    )
    device = accelerator.device

    batch_size = args.batch_size
    lr_decay = 0.95
    lr = args.lr
    lr_min = args.lr_min

    n_epoch = args.epochs

    # checkpoint path
    checkpoint_dir = args.ckpt_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # output path
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # logs path
    logs_dir = args.logs_dir
    if accelerator.is_main_process:
        # if os.path.exists(logs_dir):
        #     shutil.rmtree(logs_dir)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # dataset and dataloader
    data_set = CustomDataset(root_dir=args.train_data, transform=transforms.ToTensor(), train=True)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("Length of the data_loader :", len(data_loader))

    # model here
    model = My_model(in_chans= 64, 
        embed_dim= 60, 
        dim= 48, 
        num_blocks=[2, 2, 2, 2], 
        num_refinement_blocks= 2, 
        heads=[1, 2, 4, 8], 
        ffn_expansion_factor = 2.66, 
        bias=False,LayerNorm_type='BiasFree',
        use_checkpoint=args.use_checkpoint)
    print('\n-------Training started -------\n')

    model.train()
    model.to(accelerator.device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=lr_decay)

    average_loss = MovingAverage(200)
    if not restart_train:
        try:
            checkpoint = load_checkpoint(checkpoint_dir, 'best')
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_iter']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
        except:
            start_epoch = 0
            global_step = 0
            best_loss = np.inf
            print('=> no checkpoint file to be loaded.')
    else:
        start_epoch = 0
        global_step = 0
        best_loss = np.inf
        if os.path.exists(checkpoint_dir):
            pass
        else:
            os.mkdir(checkpoint_dir)
        print('=> training')

    MSE_loss = nn.MSELoss()
    
    # prepare Accelerator
    model, optimizer, data_loader, scheduler = accelerator.prepare(
        model, optimizer, data_loader, scheduler
    )

    for epoch in range(start_epoch, n_epoch):
        epoch_start_time = time.time()
        if accelerator.is_main_process:
            print('='*20, 'lr={}'.format([param['lr'] for param in optimizer.param_groups]), '='*20)
        avg_loss = 0
        avg_psnr = 0
        avg_ssim = 0
        avg_lpips = 0
        avg_step = 0
        for step, (burst_noise, gt) in enumerate(data_loader):
            t0 = time.time()
            # if cuda:
            #     burst_noise = burst_noise.cuda()
            #     gt = gt.cuda()
            burst_noise = burst_noise.squeeze(2)
            pred = model(burst_noise)
            loss, ssim_val, lpips_val = custom_loss(pred, gt)
            optimizer.zero_grad()
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            average_loss.update(loss)
            # pdb.set_trace()
            psnr = calculate_psnr(pred.unsqueeze(1), gt.unsqueeze(1))
            ssim = calculate_ssim(pred.unsqueeze(1), gt.unsqueeze(1))
            lpips = lpips_val.item()
            avg_loss += loss.item()
            avg_psnr += psnr
            avg_ssim += ssim
            avg_lpips += lpips
            avg_step += 1
            t1 = time.time()
            # save images
            if accelerator.is_main_process:
                if (epoch % 10 == 0) and (step < 20):
                    for frame in range(7):#9
                        pil_image = to_pil_image(burst_noise[0][frame])
                        pil_image.save(f'./{output_dir}/Batch{step}_input{frame}.png')
                    pil_image = to_pil_image(gt[0])
                    pil_image.save(f'./{output_dir}/Batch{step}_gt.png')
                    pil_image = to_pil_image(pred[0])
                    pil_image.save(f'./{output_dir}/Batch{step}_output_E{epoch}.png')
                # print
                if step % 200 == 0:
                    print('{:-4d}\t| epoch {:2d}\t| step {:4d}\t|'
                        ' loss: {:.4f}\t| PSNR: {:.2f}dB\t| SSIM: {:.4f}\t| LPIPS: {:.4f}\t| time:{:.2f} seconds.'
                        .format(global_step, epoch, step, loss, psnr, ssim, lpips,t1-t0))
            global_step += 1
        if accelerator.is_main_process:
            print('Epoch {} is finished, time elapsed {:.2f} seconds.'.format(epoch, time.time() - epoch_start_time))
            print('Average loss : {:.5f}\t| Average PSNR : {:.3f}\t| Average SSIM : {:.3f}\t| Average LPIPS : {:.4f} \n'.format(avg_loss/avg_step, avg_psnr/avg_step, avg_ssim/avg_step, avg_lpips/avg_step))
            if epoch % 10 == 0 or (epoch <= 150 and epoch % 1 == 0):
                if average_loss.get_value() < best_loss:
                    is_best = True
                    best_loss = average_loss.get_value()
                else:
                    is_best = False

                unwrapped_model = accelerator.unwrap_model(model)
                state_dict = accelerator.get_state_dict(model)
                save_dict = {
                    'epoch': epoch,
                    'global_iter': global_step,
                    'state_dict': unwrapped_model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict()
                }
                save_checkpoint(
                    save_dict, is_best, checkpoint_dir, global_step, max_keep=100 
                )


        # decay the learning rate
        lr_cur = [param['lr'] for param in optimizer.param_groups]
        if lr_cur[0] > lr_min:
            scheduler.step()
        else:
            for param in optimizer.param_groups:
                param['lr'] = lr_min



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="模型训练脚本")

    ########### --- file paths --- ##########
    parser.add_argument("--train_data", type=str, default="./dataset/train/crop_size256_stride128",
                                help="离线分块后的训练集路径")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoint",
                                help="权重保存路径")
    parser.add_argument("--output_dir", type=str, default="./output",
                                help="中间结果保存路径")
    parser.add_argument("--logs_dir", type=str, default="./logs_dir",
                                help="日志文件保存路径")

    ########### --- hyperparameters of training phase --- ##########

    parser.add_argument("--epochs", type=int, default=100, 
                                help="总训练epoch数")
    parser.add_argument("--batch_size", type=int, default=1,
                                help="单卡batch size数")
    parser.add_argument("--lr", type=float, default=2e-4, 
                                help="初始学习率")
    parser.add_argument("--lr_min", type=float, default=5e-6,
                                help="最小学习率限制")
    parser.add_argument("--num_workers", type=int, default=4, 
                                help="DataLoader进程数")

    ########### --- state control --- ##########

    parser.add_argument("--restart", action="store_true", 
                                help="是否重新开始训练")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
                                help="是否开启混合精度模式")
    parser.add_argument("--use_checkpoint", action="store_true", 
                                help="是否开启梯度检查点")
    
    args = parser.parse_args()

    train(args, num_threads=1, cuda=True, restart_train=args.restart, mGPU=4)
