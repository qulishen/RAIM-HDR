import os
import time
import logging
import itertools
import math
import json
import numpy as np
import random
from PIL import Image
import importlib
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import importlib
import sys
sys.path.append("..")
from dataloader.dataset import TrainSet #, TestSet , TestSet_multi, Urban100, Sun80
from utils.util import rss_coil_combination
from utils import util, calculate_PSNR_SSIM
from models.modules import define_G
from dataloader import DistIterSampler, create_dataloader
from skimage.metrics import peak_signal_noise_ratio,structural_similarity,normalized_root_mse
from skimage.restoration import estimate_sigma
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from diffusers import ScoreSdeVeScheduler, DDPMScheduler 
from models.diffusion_utils import *
from models.ema import ExponentialMovingAverage



def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def overlapping_grid_indices(inp, output_size, r=None):
    """
    Generate overlapping grid indices for patch-based processing.
    """
    _, c, h, w = inp.shape
    r = 16 if r is None else r
    h_list = [i for i in range(0, h - output_size + 1, r)]
    w_list = [i for i in range(0, w - output_size + 1, r)]
    
    # Make sure we cover the entire image by adding the final index if needed
    if h - output_size not in h_list:
        h_list.append(h - output_size)
    if w - output_size not in w_list:
        w_list.append(w - output_size)
        
    return h_list, w_list

def unpatchify_restore_overlapping(x, model=None, corners=None, p_size=64, time=None, model_time_conditioning=False):
    """
    Restore image by processing overlapping patches with time conditioning support.
    
    Args:
        x: Input tensor
        model: Neural network model
        corners: List of (h, w) coordinates for patches
        p_size: Patch size
        time: Time parameter for conditioning (noise level)
        model_time_conditioning: Whether model uses time conditioning
    """
    with torch.no_grad():
        x_grid_mask = torch.zeros_like(x, device=x.device)
        e_output = torch.zeros_like(x, device=x.device)
        
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1
            
        for (hi, wi) in corners:
            x_patch = x[:, :, hi:hi+p_size, wi:wi+p_size]
            if model is None:
                e_output[:, :, hi:hi + p_size, wi:wi + p_size] += x_patch
            else:
                if model_time_conditioning and time is not None:
                    num_channels = x_patch.shape[1]
                    is_color = (num_channels == 3)
                    sigma_values = torch.zeros(x_patch.shape[0]).to(x_patch.device)
                    # Iterate over the batch to estimate sigma for each image
                    for i in range(x_patch.shape[0]):
                        if is_color:
                            # For color images, estimate sigma for each channel separately and average
                            channel_sigmas = []
                            for c in range(num_channels):
                                img_channel = x_patch[i, c, :, :].detach().cpu().numpy()
                                channel_sigmas.append(estimate_sigma(img_channel))
                            # Average sigma across channels
                            sigma_values[i] = torch.tensor(np.mean(channel_sigmas), device=x_patch.device)
                        else:
                            # For grayscale, estimate sigma on the single channel
                            img = x_patch[i, 0, :, :].detach().cpu().numpy()
                            sigma_values[i] = estimate_sigma(img)
                    noise_amount = sigma_values                            

                    # Use time conditioning if enabled
                    output_patch = model(x1=x_patch, time=noise_amount)
                else:
                    # Regular model forward pass
                    output_patch = model(x_patch)
                e_output[:, :, hi:hi + p_size, wi:wi + p_size] += output_patch
                
        # Average the overlapping regions
        e = torch.div(e_output, x_grid_mask)
    return e

def patch_based_restoration(x, model=None, r=None, patch_size=64, time=None, model_time_conditioning=False):
    """
    Perform patch-based restoration with time conditioning support.
    
    Args:
        x: Input tensor
        model: Neural network model
        r: Stride for overlapping patches
        patch_size: Size of each patch
        time: Time parameter for conditioning (noise level)
        model_time_conditioning: Whether model uses time conditioning
    """
    p_size = patch_size
    h_list, w_list = overlapping_grid_indices(x, output_size=p_size, r=r)
    corners = [(i, j) for i in h_list for j in w_list]
    x_output = unpatchify_restore_overlapping(
        x, 
        model=model, 
        corners=corners, 
        p_size=p_size, 
        time=time,
        model_time_conditioning=model_time_conditioning
    )
    return x_output





class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.augmentation = args.data_augmentation
        self.device = torch.device('cuda' if len(args.gpu_ids) != 0 else 'cpu')
        self.num_timesteps = args.timesteps
        self.normalize = normalize_to_neg_one_to_one if args.auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if args.auto_normalize else identity
        args.device = self.device
        if args.beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif args.beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif args.beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {args.beta_schedule}')

        betas = beta_schedule_fn(self.num_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        self.alphas_cumprod = alphas_cumprod.to(torch.float32)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(torch.float32)
        self.betas = betas.to(torch.float32)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(torch.float32)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).to(torch.float32)
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).to(torch.float32)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).to(torch.float32)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).to(torch.float32)

        # for ve sde
        self.ve_noise_scheduler = ScoreSdeVeScheduler(num_train_timesteps=self.num_timesteps, sigma_min=self.args.ve_sde_sigma_min, sigma_max=self.args.ve_sde_sigma_max)

        # # for ema
        self.use_ema = True if args.ema else False
        self.ema_decay = args.ema_decay
        self.best_metric_value = -float('inf')
        self.best_metric_name = getattr(args, 'best_metric', 'proxy')
        self.eval_lpips = bool(getattr(args, 'eval_lpips', False) or self.best_metric_name == 'proxy')

        # dynamically load dataset based on args.modal / multi-contrast
        if args.dataset == 'M4Raw':
            ## init dataloader
            if args.phase == 'train':
                trainset_ = getattr(importlib.import_module('dataloader.dataset'), args.trainset, None)
                if  hasattr(args, 'modal') and args.modal == 'ALL':
                    self.args.modal = 'T1'
                    self.train_dataset1 = trainset_(self.args)
                    self.args.modal = 'T2'
                    self.train_dataset2 = trainset_(self.args)
                    self.args.modal = 'FLAIR'
                    self.train_dataset3 = trainset_(self.args)
                    self.train_dataset = self.train_dataset1 + self.train_dataset2 + self.train_dataset3
                else:
                    self.train_dataset = trainset_(self.args)
                if args.dist:
                    dataset_ratio = 1
                    train_sampler = DistIterSampler(self.train_dataset, args.world_size, args.rank, dataset_ratio)
                    self.train_dataloader = create_dataloader(self.train_dataset, args, train_sampler)
                else:
                    self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
            testset_ = getattr(importlib.import_module('dataloader.dataset'), args.testset, None)
        elif 'fastMRI' in args.dataset:
            if args.phase == 'train':
                trainset_ = getattr(importlib.import_module('dataloader.dataset'), args.trainset, None)
                self.train_dataset = trainset_(self.args)
                if args.dist:
                    dataset_ratio = 1
                    train_sampler = DistIterSampler(self.train_dataset, args.world_size, args.rank, dataset_ratio)
                    self.train_dataloader = create_dataloader(self.train_dataset, args, train_sampler)
                else:
                    self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

            testset_ = getattr(importlib.import_module('dataloader.dataset'), args.testset, None)

        self.test_dataset = testset_(self.args)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

        ## init network
        self.net = define_G(args)
        if args.resume:
            self.load_networks('net', self.args.resume)

        if args.rank <= 0:
            logging.info('----- network parameters: %f -----' % (sum(param.numel() for param in self.net.parameters()) / (10**6)))

        ## init loss and optimizer
        if args.phase == 'train':
            if args.rank <= 0:
                logging.info('init criterion and optimizer...')
            g_params = [self.net.parameters()]

            self.criterion_mse = nn.MSELoss().to(self.device)
            if args.loss_mse:
                self.criterion_mse = nn.MSELoss().to(self.device)
                self.lambda_mse = args.lambda_mse
                if args.rank <= 0:
                    logging.info('  using mse loss...')

            if args.loss_n2s:
                self.criterion_n2s = nn.MSELoss().to(self.device)
                self.lambda_n2s = args.lambda_n2s
                if args.rank <= 0:
                    logging.info('  using n2s loss...')

            if args.loss_c2s_sc_model:
                self.criterion_c2s_sc_model = nn.MSELoss().to(self.device)
                self.lambda_c2s_sc_model = args.lambda_c2s_sc_model
                if args.rank <= 0:
                    logging.info('  using loss_c2s_sc_model loss...')

            if args.loss_d2s_sft:
                self.criterion_d2s_sft = nn.MSELoss().to(self.device)
                self.lambda_d2s_sft = args.lambda_d2s_sft
                if args.rank <= 0:
                    logging.info('  using loss_d2s_sft loss...')

            if args.loss_d2s_pretrain:
                self.criterion_d2s_pretrain = nn.MSELoss().to(self.device)
                self.lambda_d2s_pretrain = args.lambda_d2s_pretrain
                if args.rank <= 0:
                    logging.info('  using loss_d2s_pretrain loss...')

            if args.loss_mdae:
                self.criterion_mdae = nn.MSELoss(reduction='none').to(self.device)
                self.lambda_mdae = args.lambda_mdae
                self.lambda_mdae_masked = args.lambda_mdae_masked
                self.lambda_mdae_visible = args.lambda_mdae_visible
                if args.rank <= 0:
                    logging.info('  using loss_mdae (MDAE) loss...')

            if args.loss_r2r:
                self.criterion_r2r = nn.MSELoss().to(self.device)
                self.lambda_r2r = args.lambda_r2r
                self.eps_r2r = args.eps_r2r
                self.alpha_r2r = args.alpha_r2r
                if args.rank <= 0:
                    logging.info('  using r2r loss...')

            if args.loss_l1:
                self.criterion_l1 = nn.L1Loss().to(self.device)
                self.lambda_l1 = args.lambda_l1
                if args.rank <= 0:
                    logging.info('  using l1 loss...')

            if args.loss_noise2score:
                self.lambda_noise2score_loss = args.lambda_noise2score_loss
                if args.rank <= 0:
                    logging.info('  using noise2score_loss...')

            if args.loss_composite or self.eval_lpips:
                self.criterion_l1_comp = nn.L1Loss().to(self.device)
                from pytorch_msssim import ssim as ssim_fn
                self.ssim_fn = ssim_fn
                import lpips
                self.criterion_lpips = lpips.LPIPS(net='alex').to(self.device)
                for p in self.criterion_lpips.parameters():
                    p.requires_grad = False
                if args.rank <= 0 and args.loss_composite:
                    logging.info(
                        '  using composite loss (L1=%.2f + SSIM=%.2f + LPIPS=%.2f + GRAD=%.2f + SAT=%.2f)...'
                        % (
                            args.lambda_composite_l1,
                            args.lambda_ssim,
                            args.lambda_lpips,
                            args.lambda_grad,
                            args.lambda_sat,
                        )
                    )


            g_params = [self.net.parameters()]
            self.optimizer_G = AdamW(itertools.chain.from_iterable(g_params), lr=args.lr, weight_decay=args.weight_decay)

            self.scheduler = CosineAnnealingLR(self.optimizer_G, T_max=args.max_iter)

            if args.resume_optim:
                self.load_networks('optimizer_G', self.args.resume_optim)
            if args.resume_scheduler:
                self.load_networks('scheduler', self.args.resume_scheduler)


    def vis_results(self, epoch, i, images):
        for j in range(min(images[0].size(0), 2)):
            if self.args.net_3D:
                # Selecting the middle slice from the depth dimension [batch, channels, depth, height, width]
                slice_index = images[0].size(2) // 2
                images_slice = [img[j, :, slice_index, :, :] for img in images]

            elif self.args.output_nc == 2:
                images_slice = [img[j] for img in images]
            else:
                # For multi-channel input (e.g. MEF 15ch), extract middle 3 channels for vis
                in_ch = getattr(self.args, 'in_channels', 3)
                images_slice = []
                for img in images:
                    t = img[j]
                    if t.shape[0] > 3 and t.shape[0] == in_ch:
                        # Take middle exposure (channels 6:9 for 5-frame MEF)
                        mid = (t.shape[0] // 3) // 2 * 3
                        t = t[mid:mid+3]
                    images_slice.append(t)

            save_name = os.path.join(self.args.vis_save_dir, 'vis_%d_%d_%d.jpg' % (epoch, i, j))
            temps = torch.stack(images_slice)
            torchvision.utils.save_image(temps, save_name)

    def vis_test_results(self, i, images):
        for j in range(min(images[0].size(0), 2)):
            if self.args.net_3D:
                # Selecting the middle slice from the depth dimension
                slice_index = images[0].size(2) // 2
                images_slice = [img[j, :, slice_index, :, :] for img in images]
            elif self.args.output_nc == 2:
                images_slice = [img[j] for img in images]
            else:
                # For multi-channel input, extract middle 3 channels for vis
                in_ch = getattr(self.args, 'in_channels', 3)
                images_slice = []
                for img in images:
                    t = img[j]
                    if t.shape[0] > 3 and t.shape[0] == in_ch:
                        mid = (t.shape[0] // 3) // 2 * 3
                        t = t[mid:mid+3]
                    images_slice.append(t)

            save_name = os.path.join(self.args.save_folder, 'test_vis_%d_%d.jpg' % (i, j))
            temps = torch.stack(images_slice)
            torchvision.utils.save_image(temps, save_name)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def prepare(self, batch_samples):
        for key in batch_samples.keys():
            if 'name' not in key and 'pad_nums' not in key:
                batch_samples[key] = Variable(batch_samples[key].to(self.device), requires_grad=False)
        return batch_samples

    def estimate_conditioning_sigma(self, normalized_input):
        """Estimate the time-conditioning scalar used by TimeDiffiT."""
        batch_size, num_channels = normalized_input.shape[:2]
        sigma = torch.zeros(batch_size, device=normalized_input.device)

        # For stacked MEF inputs, use the middle exposure statistics.
        if num_channels > 3 and num_channels % 3 == 0:
            mid_start = (num_channels // 3) // 2 * 3
            for idx in range(batch_size):
                sigma[idx] = normalized_input[idx, mid_start:mid_start + 3].std()
            return torch.clamp(sigma, min=0.01)

        if self.args.estimate_sigma:
            is_color = (num_channels == 3)
            for idx in range(batch_size):
                if is_color:
                    channel_sigmas = []
                    for channel in range(num_channels):
                        img_channel = normalized_input[idx, channel].detach().cpu().numpy()
                        channel_sigmas.append(estimate_sigma(img_channel))
                    sigma[idx] = torch.tensor(np.mean(channel_sigmas), device=normalized_input.device)
                else:
                    img = normalized_input[idx, 0].detach().cpu().numpy()
                    sigma[idx] = torch.tensor(estimate_sigma(img), device=normalized_input.device)
            return torch.clamp(sigma, min=0.01)

        if self.args.noise_model:
            base_sigma = 2 * self.args.noise_std if self.args.auto_normalize else self.args.noise_std
            return torch.ones(batch_size, device=normalized_input.device) * base_sigma

        return sigma

    def compute_gradient_loss(self, prediction, target):
        pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
        pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

    def compute_saturation_mask(self, input_stack):
        if input_stack.shape[1] <= 3 or input_stack.shape[1] % 3 != 0:
            return torch.ones(input_stack.shape[0], 1, input_stack.shape[2], input_stack.shape[3], device=input_stack.device)

        frames = input_stack.reshape(input_stack.shape[0], input_stack.shape[1] // 3, 3, input_stack.shape[2], input_stack.shape[3])
        over_mask = (frames >= self.args.sat_threshold_high).any(dim=2).any(dim=1, keepdim=True)
        under_mask = (frames <= self.args.sat_threshold_low).any(dim=2).any(dim=1, keepdim=True)
        return (over_mask | under_mask).float()

    def compute_proxy_score(self, psnr, ssim, lpips_value):
        return 30.0 * (psnr / 50.0) + 22.5 * ((ssim - 0.5) / 0.5) + 30.0 * (1.0 - (lpips_value / 0.4))

    def save_validation_metrics(self, epoch, metrics, is_best=False):
        if self.args.rank > 0:
            return

        latest_path = os.path.join(self.args.save_folder, 'val_metrics_latest.json')
        with open(latest_path, 'w') as f:
            json.dump(metrics, f, indent=2, sort_keys=True)

        history_path = os.path.join(self.args.save_folder, 'val_metrics_history.jsonl')
        with open(history_path, 'a') as f:
            f.write(json.dumps(metrics, sort_keys=True) + '\n')

        if is_best:
            best_path = os.path.join(self.args.save_folder, 'val_metrics_best.json')
            with open(best_path, 'w') as f:
                json.dump(metrics, f, indent=2, sort_keys=True)


    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod.to(self.device), t.to(self.device), x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod.to(self.device), t.to(self.device), x_start.shape) * noise
        )

    def train(self):
        if self.args.rank <= 0:
            logging.info('training on  ...' + self.args.dataset)
            logging.info('%d training samples' % (self.train_dataset.__len__()))
            logging.info('the init lr: %f'%(self.args.lr))
        steps = 0
        self.net.train()
        if self.use_ema:
            ema = ExponentialMovingAverage(self.net.parameters(), decay=self.ema_decay)
            self.state = dict(optimizer=self.optimizer_G, model=self.net, ema=ema, step=0)

        if self.args.use_tb_logger:
            if self.args.rank <= 0:
                tb_logger = SummaryWriter(log_dir='/scratch/jtu9/tb_logger/' + self.args.name)

        self.augmentation = False  # disenable data augmentation to warm up the encoder
        # epoch-based training
        for i in range(self.args.start_iter, self.args.max_iter):
            self.scheduler.step()
            logging.info('current_lr: %f' % (self.optimizer_G.param_groups[0]['lr']))
            t0 = time.time()
            if self.args.loss_noise2score:
                epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
            # training iter per epoch
            for j, batch_samples in enumerate(self.train_dataloader):
                log_info = 'epoch:%03d step:%04d  ' % (i, j)

                ## prepare data
                batch_samples = self.prepare(batch_samples)
                if 'labels' in batch_samples:
                    labels = batch_samples['labels']
                else:
                    labels = batch_samples['plabels']
                if self.args.loss_n2s:
                    net_input = batch_samples['masked_images']
                else:
                    net_input = batch_samples['images']


                ## optimization
                loss = 0
                self.optimizer_G.zero_grad()

                if self.args.model_time_conditioning == False and self.args.loss_c2s_sc_model == False:
                    output = self.net(net_input)

                if self.args.loss_mse:
                    mse_loss = self.criterion_mse(output,labels)
                    mse_loss = mse_loss * self.lambda_mse
                    loss += mse_loss
                    log_info += 'mse_loss:%.06f ' % (mse_loss.item())



                if self.args.loss_n2s:
                    mask_real = batch_samples['mask_real']
                    n2s_loss = self.criterion_n2s(output*mask_real,labels) # mse on masked pixels, labels already masked before batching
                    n2s_loss = n2s_loss * self.lambda_n2s
                    loss += n2s_loss
                    log_info += 'n2s_loss:%.06f ' % (n2s_loss.item())

                if self.args.loss_r2r:
                    eps_r2r = self.eps_r2r
                    alpha_r2r = self.alpha_r2r
                    pert = eps_r2r*torch.FloatTensor(net_input.size()).normal_(mean=0, std=1.).cuda()
                    input_train = net_input + alpha_r2r*pert
                    output = self.net(input_train)
                    labels = net_input - pert/alpha_r2r
                    r2r_loss = self.criterion_r2r(output,labels)
                    r2r_loss = r2r_loss * self.lambda_r2r
                    loss += r2r_loss
                    log_info += 'r2r_loss:%.06f ' % (r2r_loss.item())

                # supervised finetuning of a pretrained C2S/D2S model -- || f(X_t_data, t_data) <-> label ||^2
                if self.args.loss_d2s_sft:
                    # mask_real = batch_samples['mask_real']
                    x1 = net_input
                    x1 = self.normalize(x1)
                    target = self.normalize(labels)
 
                    # Get number of channels
                    num_channels = x1.shape[1]
                    is_color = (num_channels == 3)


                    if self.args.estimate_sigma:
                        sigma_0 = torch.zeros(x1.shape[0]).to(x1.device)
                        
                        for _i in range(x1.shape[0]):
                            if is_color:
                                # For color images, estimate sigma for each channel separately and average
                                channel_sigmas = []
                                for c in range(num_channels):
                                    img_channel = x1[_i, c, :, :].detach().cpu().numpy()
                                    channel_sigmas.append(estimate_sigma(img_channel))
                                # Average sigma across channels
                                sigma_0[_i] = torch.tensor(np.mean(channel_sigmas), device=x1.device)
                            else:
                                # For grayscale, estimate sigma on the single channel
                                img = x1[_i, 0, :, :].detach().cpu().numpy()
                                sigma_0[_i] = estimate_sigma(img)
                    else:
                        # Convert scalar noise_std to a tensor of appropriate shape
                        sigma_0 = torch.ones(x1.shape[0]).to(x1.device) * self.args.noise_std * 2  # due to normalization to [-1,1]

                    sigma_t_data = sigma_0 # t_data 

                    if self.args.model_time_conditioning == False:
                        output = self.net(x1) 
                    else:
                        output = self.net(x1 = x1, time = sigma_t_data)

                    d2s_sft_loss = self.criterion_d2s_sft(output, target)
                    d2s_sft_loss = d2s_sft_loss * self.lambda_d2s_sft
                    loss += d2s_sft_loss
                    log_info += 'd2s_sft_loss:%.06f ' % (d2s_sft_loss.item())


                # Composite supervised loss (L1 + SSIM + LPIPS)
                if self.args.loss_composite:
                    x_comp = self.normalize(net_input) if self.args.auto_normalize else net_input
                    target_comp = self.normalize(labels) if self.args.auto_normalize else labels

                    # Forward pass with the same MEF-aware time conditioning used at inference.
                    if self.args.model_time_conditioning:
                        sigma_comp = self.estimate_conditioning_sigma(x_comp)
                        output = self.net(x1=x_comp, time=sigma_comp)
                    else:
                        output = self.net(x_comp)

                    # L1 loss
                    comp_l1 = self.criterion_l1_comp(output, target_comp) * self.args.lambda_composite_l1
                    loss += comp_l1

                    # SSIM loss (needs [0,1] range)
                    out_01 = torch.clamp((output + 1) / 2, 0, 1) if self.args.auto_normalize else torch.clamp(output, 0, 1)
                    tgt_01 = (target_comp + 1) / 2 if self.args.auto_normalize else target_comp
                    ssim_val = self.ssim_fn(out_01, tgt_01, data_range=1.0, size_average=True)
                    comp_ssim = (1 - ssim_val) * self.args.lambda_ssim
                    loss += comp_ssim

                    # LPIPS loss (needs [-1,1] range)
                    lpips_in = output if self.args.auto_normalize else output * 2 - 1
                    lpips_tgt = target_comp if self.args.auto_normalize else target_comp * 2 - 1
                    comp_lpips = self.criterion_lpips(lpips_in, lpips_tgt).mean() * self.args.lambda_lpips
                    loss += comp_lpips

                    comp_grad = output.new_tensor(0.0)
                    if getattr(self.args, 'lambda_grad', 0.0) > 0:
                        comp_grad = self.compute_gradient_loss(out_01, tgt_01) * self.args.lambda_grad
                        loss += comp_grad

                    comp_sat = output.new_tensor(0.0)
                    if getattr(self.args, 'lambda_sat', 0.0) > 0:
                        sat_mask = self.compute_saturation_mask(net_input)
                        masked_error = torch.abs(out_01 - tgt_01) * sat_mask
                        denom = sat_mask.sum() * out_01.shape[1]
                        comp_sat = masked_error.sum() / denom.clamp(min=1.0)
                        comp_sat = comp_sat * self.args.lambda_sat
                        loss += comp_sat

                    log_info += 'L1:%.4f SSIM:%.4f(%.4f) LPIPS:%.4f GRAD:%.4f SAT:%.4f ' % (
                        comp_l1.item(),
                        comp_ssim.item(),
                        ssim_val.item(),
                        comp_lpips.item(),
                        comp_grad.item(),
                        comp_sat.item(),
                    )

                # pretrain from clean images -- <corrupted clean images -- self>
                if self.args.loss_d2s_pretrain:
                    # mask_real = batch_samples['mask_real']
                    x1 = labels # corrupt the clean image 
                    x1 = self.normalize(x1)
                    target = self.normalize(labels)
 
                    # Get number of channels
                    num_channels = x1.shape[1]
                    noise_amount = torch.rand(x1.shape[0]).to(self.device)
                    noisy_x1 = corrupt(x1, noise_amount, sde=self.args.sde, max_sigma=self.args.ve_sde_sigma_max) # Create noisy x1


                    if self.args.sde == 've':                    
                        var_input =(noise_amount * self.args.ve_sde_sigma_max)**2
                    else:
                        var_input = noise_amount * 1 # resulting noise variance for ddpm corruption
                    sigma_input = torch.sqrt(var_input)

                    if self.args.model_time_conditioning == False:
                        output = self.net(noisy_x1) 
                    else:
                        output = self.net(x1 = noisy_x1, time = sigma_input) 


                    d2s_pretrain_loss = self.criterion_d2s_pretrain(output, target)
                    d2s_pretrain_loss = d2s_pretrain_loss * self.lambda_d2s_pretrain
                    loss += d2s_pretrain_loss
                    log_info += 'd2s_pretrain_loss:%.06f ' % (d2s_pretrain_loss.item())


                if self.args.loss_mdae:
                    # Start from clean labels (pretraining loss)
                    x1 = labels
                    x1 = self.normalize(x1)
                    target = self.normalize(labels)

                    B, C, H, W = x1.shape

                    # Random masking percentage per batch from U(min, max)
                    mask_pct = random.uniform(self.args.mdae_mask_pct_min, self.args.mdae_mask_pct_max)

                    # Create block mask: 1=visible, 0=masked
                    spatial_mask = create_block_mask_2d(
                        B, H, W, mask_pct, block_size=self.args.mdae_block_size
                    ).to(self.device)

                    # Mask the image (zero out masked regions)
                    masked_input = x1 * spatial_mask

                    # Corrupt visible regions with random noise
                    noise_amount = torch.rand(B).to(self.device)
                    corrupted_input = corrupt(masked_input, noise_amount, sde=self.args.sde,
                                              max_sigma=self.args.ve_sde_sigma_max)
                    # Re-zero masked regions after corruption
                    network_input = corrupted_input * spatial_mask

                    # Compute sigma_t for time conditioning
                    if self.args.sde == 've':
                        sigma_t = noise_amount * self.args.ve_sde_sigma_max
                    else:
                        sigma_t = torch.sqrt(noise_amount)

                    # Forward pass
                    if self.args.model_time_conditioning:
                        output = self.net(x1=network_input, time=sigma_t)
                    else:
                        output = self.net(network_input)

                    # Per-pixel MSE (reduction='none')
                    pixel_loss = self.criterion_mdae(output, target)  # [B, C, H, W]

                    # Separate losses for masked vs visible regions
                    masked_region = (1 - spatial_mask)  # 1 where masked
                    visible_region = spatial_mask       # 1 where visible

                    masked_count = masked_region.sum().clamp(min=1)
                    visible_count = visible_region.sum().clamp(min=1)

                    masked_loss = (pixel_loss * masked_region).sum() / masked_count
                    visible_loss = (pixel_loss * visible_region).sum() / visible_count

                    mdae_loss = (self.lambda_mdae_masked * masked_loss +
                                 self.lambda_mdae_visible * visible_loss) * self.lambda_mdae

                    loss += mdae_loss
                    log_info += 'mdae_loss:%.06f mdae_masked:%.06f mdae_visible:%.06f mask_pct:%.02f ' % (
                        mdae_loss.item(), masked_loss.item(), visible_loss.item(), mask_pct)


                if self.args.loss_c2s_sc_model:
                    # mask_real = batch_samples['mask_real']
                    x1 = net_input
                    x1 = self.normalize(x1)
                    # target = x1 * mask_real
                    target = x1 
                    
                    # Get number of channels
                    num_channels = x1.shape[1]
                    is_color = (num_channels == 3)

                    if self.args.loss_noiser2noise:
                        noise_amount = torch.zeros(x1.shape[0]).to(self.device) + self.args.alpha_noiser2noise
                        noisy_x1 = corrupt(x1, noise_amount, sde=self.args.sde, max_sigma=self.args.ve_sde_sigma_max) # Create noisy x1
                    else:
                        if not self.args.use_noreparam:
                            noise_amount = torch.rand(x1.shape[0]).to(self.device)
                            noisy_x1 = corrupt(x1, noise_amount, sde=self.args.sde, max_sigma=self.args.ve_sde_sigma_max) # Create noisy x1


                    if self.args.estimate_sigma:
                        sigma_0 = torch.zeros(x1.shape[0]).to(x1.device)
                        
                        for _i in range(x1.shape[0]):
                            if is_color:
                                # For color images, estimate sigma for each channel separately and average
                                channel_sigmas = []
                                for c in range(num_channels):
                                    img_channel = x1[_i, c, :, :].detach().cpu().numpy()
                                    channel_sigmas.append(estimate_sigma(img_channel))
                                # Average sigma across channels
                                sigma_0[_i] = torch.tensor(np.mean(channel_sigmas), device=x1.device)
                            else:
                                # For grayscale, estimate sigma on the single channel
                                img = x1[_i, 0, :, :].detach().cpu().numpy()
                                sigma_0[_i] = estimate_sigma(img)
                    else:
                        # Convert scalar noise_std to a tensor of appropriate shape
                        sigma_0 = torch.ones(x1.shape[0]).to(x1.device) * self.args.noise_std * 2  # due to normalization to [-1,1]

                    if self.args.use_noreparam:
                        # random sample sigma_t from uniform distribution in the range of (sigma_0, self.args.ve_sde_sigma_max] of the shape x1.shape[0]
                        sigma_t = torch.rand(x1.shape[0]).to(x1.device) * (self.args.ve_sde_sigma_max - sigma_0) + sigma_0
                        # calculate sigma_t_prime = torch.sqrt(sigma_t**2 - sigma_0**2)
                        sigma_t_prime = torch.sqrt(sigma_t**2 - sigma_0**2)
                        sigma_t_prime_noise_amount = sigma_t_prime / self.args.ve_sde_sigma_max
                        noisy_x1 = corrupt(x1, sigma_t_prime_noise_amount, sde=self.args.sde, max_sigma=self.args.ve_sde_sigma_max) # Create noisy x1

                    if not self.args.use_noreparam:
                        if self.args.sde == 've':                    
                            var_input = sigma_0**2 + (noise_amount * self.args.ve_sde_sigma_max)**2
                        else:
                            var_input = (1-noise_amount) * sigma_0**2 + noise_amount * 1 # resulting noise variance for ddpm corruption
                        sigma_input = torch.sqrt(var_input)

                    if self.args.model_time_conditioning == False:
                        # combined_input = noisy_x1 * (1-mask_real) + noise * mask_real
                        combined_input = noisy_x1 
                        output = self.net(combined_input) 
                    else:
                        # combined_input = noisy_x1 * (1-mask_real) + noise * mask_real
                        combined_input = noisy_x1 
 
                        if not self.args.use_noreparam:
                            output = self.net(x1 = combined_input, time = sigma_input) 
                            noise_amount = noise_amount.view(-1, 1, 1, 1).expand_as(noisy_x1)
                            sigma_0 = sigma_0.view(-1, 1, 1, 1).expand_as(noisy_x1)

                        else:
                            output = self.net(x1 = combined_input, time = sigma_t)



                    if self.args.ambient:
                        target_snr_ratio_stage_1 = 0  # MAXIMUM Denoising for stage I 

                        if not self.args.use_noreparam:
                            t_sample = noise_amount * self.args.ve_sde_sigma_max if self.args.sde == 've' else noise_amount # noise amount directly for ddpm corruption, else needs to consider sigma_max

                            # ve
                            if self.args.sde == 've':
                                output_stage_1 = (output - noisy_x1) * (t_sample**2) / (t_sample**2 + sigma_0**2 - (sigma_0 * target_snr_ratio_stage_1)**2) + noisy_x1 # aug 31
                            else:
                                # output = (output) * (t_sample) / (t_sample + sigma_0**2 - t_sample * sigma_0**2) + noisy_x1 * (torch.sqrt(1 - t_sample)) / ((t_sample / sigma_0**2) - t_sample + 1) # t_sample refers to the sigma_t' **2 as noise amount is defined as sigma_t' **2
                                output_stage_1 = (output) * (torch.sqrt(1 - sigma_0**2)) * (t_sample) / (t_sample + sigma_0**2 - t_sample * sigma_0**2) + noisy_x1 * (torch.sqrt(1 - t_sample)) / ((t_sample / sigma_0**2) - t_sample + 1) # t_sample refers to the sigma_t' **2 as noise amount is defined as sigma_t' **2
                        else:
                            sigma_0 = sigma_0.view(-1, 1, 1, 1).expand_as(noisy_x1)
                            sigma_t = sigma_t.view(-1, 1, 1, 1).expand_as(noisy_x1)
                            output_stage_1 = (output - noisy_x1) * (sigma_t**2 - sigma_0**2) / (sigma_t**2 - (sigma_0 * target_snr_ratio_stage_1)**2) + noisy_x1 # aug 31


                    if self.args.refine:  # Only execute this block if refine is True
                        if self.args.ambient:
                            if self.args.random_target_snr_ratio:
                                target_snr_ratio = random.uniform(0, self.args.target_snr_ratio)  # Random value between [0, self.args.target_snr_ratio]
                            else:
                                target_snr_ratio = self.args.target_snr_ratio  # Use the fixed value (e.g., 1/sqrt(2) for FLAIR and 1/sqrt(3) for T1, T2)
                            t_sample = noise_amount * self.args.ve_sde_sigma_max if self.args.sde == 've' else noise_amount # noise amount directly for ddpm corruption, else needs to consider sigma_max

                            # ve
                            if self.args.sde == 've':
                                output_stage_2 = (output - noisy_x1) * (t_sample**2) / (t_sample**2 + sigma_0**2 - (sigma_0 * target_snr_ratio)**2) + noisy_x1 # aug 31
                            else:
                                # output = (output) * (t_sample) / (t_sample + sigma_0**2 - t_sample * sigma_0**2) + noisy_x1 * (torch.sqrt(1 - t_sample)) / ((t_sample / sigma_0**2) - t_sample + 1) # t_sample refers to the sigma_t' **2 as noise amount is defined as sigma_t' **2
                                output_stage_2 = (output) * (torch.sqrt(1 - sigma_0**2)) * (t_sample) / (t_sample + sigma_0**2 - t_sample * sigma_0**2) + noisy_x1 * (torch.sqrt(1 - t_sample)) / ((t_sample / sigma_0**2) - t_sample + 1) # t_sample refers to the sigma_t' **2 as noise amount is defined as sigma_t' **2

                    if self.args.refine:
                        lambda_stage_2 = self.args.lambda_refinement
                        c2s_singlecontrast_loss = self.criterion_c2s_sc_model(output_stage_1, target) + lambda_stage_2 * self.criterion_c2s_sc_model(output_stage_2, target)
                    else:
                        c2s_singlecontrast_loss = self.criterion_c2s_sc_model(output_stage_1, target)
                    c2s_singlecontrast_loss = c2s_singlecontrast_loss * self.lambda_c2s_sc_model
                    
                    if not self.args.use_noreparam:
                        # if the self.args.sde is 've', then weight the loss with sigma**2 (noise_amount * self.args.ve_sde_sigma_max)
                        if self.args.sde == 've':
                            # amount = sigma_input.view(-1, 1, 1, 1)  # Reshape for broadcasting
                            # sigma = amount * self.args.ve_sde_sigma_max
                            sigma = sigma_input.view(-1, 1, 1, 1)  # Reshape for broadcasting
                            c2s_singlecontrast_loss = c2s_singlecontrast_loss * sigma**2
                            log_info += 've_sde_sigma_max:%.06f ' % (self.args.ve_sde_sigma_max)

                    else:
                        if self.args.sde == 've':
                            c2s_singlecontrast_loss = c2s_singlecontrast_loss * sigma_t**2
                            log_info += 've_sde_sigma_max:%.06f ' % (self.args.ve_sde_sigma_max)
                            # also log sigma_t
                            log_info += 'sigma_t mean:%.06f ' % (sigma_t.mean().item())


                    log_info += 'loss_c2s_sc_model mean:%.05f ' % (c2s_singlecontrast_loss.mean().item())
                    loss += c2s_singlecontrast_loss.mean()


                    # also log the sde used indicated by self.args.sde
                    log_info += 'sde:%s ' % (self.args.sde)



                if self.args.loss_l1: 
                    l1_loss = self.criterion_l1(output, labels)
                    l1_loss = l1_loss * self.lambda_l1
                    loss += l1_loss
                    log_info += 'l1_loss:%.06f ' % (l1_loss.item())


                if self.args.loss_noise2score:
                    epoch_iter += self.args.batch_size
                    x1 = net_input
                    
                    # Apply normalization if auto_normalize is enabled
                    if self.args.auto_normalize:
                        x1 = self.normalize(x1)
                    
                    if self.args.DSM:
                        # Denoising Score Matching approach
                        
                        # Sample random noise level -- enable negative corruption or not?
                        noise_amount = torch.randn(x1.shape[0]).to(self.device)
                        # noise_amount = torch.randn(x1.shape[0]).to(self.device) if self.args.auto_normalize else torch.rand(x1.shape[0]).to(self.device)
                        
                        # Create noisy samples using corrupt function
                        noisy_x1 = corrupt(x1, noise_amount, sde=self.args.sde, max_sigma=self.args.ve_sde_sigma_max)
                        
                        # Calculate effective sigma based on SDE type
                        if self.args.sde == 've':
                            # For VE-SDE, noise scales with ve_sde_sigma_max
                            effective_sigma = noise_amount * self.args.ve_sde_sigma_max
                        else:
                            # For other SDEs (like VP), noise amount directly represents sigma
                            effective_sigma = noise_amount
                        
                        # Calculate the implicit noise that was added
                        n = (noisy_x1 - x1)
                        
                        # Calculate target score based on the noise and effective sigma
                        effective_sigma_view = effective_sigma.view(-1, 1, 1, 1).expand_as(n)
                        dsm_target = -n / effective_sigma_view**2  # Score is gradient of log density
                        
                        # Get model prediction
                        if self.args.model_time_conditioning == False:
                            # Without time conditioning
                            score_pred = self.net(noisy_x1)
                        else:
                            # With time conditioning
                            score_pred = self.net(x1=noisy_x1, time=effective_sigma)
                        
                        # Calculate DSM loss
                        if self.args.use_dsm_weighting:
                            # Weight by sigma^2 as in the original DSM paper
                            weight = (effective_sigma_view ** 2)
                            loss_noise2score_loss = self.criterion_mse(score_pred, dsm_target) * weight
                            loss_noise2score_loss = loss_noise2score_loss.mean()
                        else:
                            loss_noise2score_loss = self.criterion_mse(score_pred, dsm_target)
                        
                    else:
                        # AR-DAE approach (original implementation)
                        
                        # Apply sigma annealing for AR-DAE
                        sigma_annealing_noise2score = self.train_dataset.__len__()
                        perc = min((epoch_iter+1)/float(sigma_annealing_noise2score), 1.0)
                        sigma_a = self.args.sigma_max_noise2score * (1-perc) + self.args.sigma_min_noise2score * perc
                        
                        # Generate noise
                        n = torch.randn_like(x1) if self.args.auto_normalize else torch.rand_like(x1)
                        
                        # Apply sigma
                        sigma_a_sample = torch.randn_like(x1) * sigma_a
                        
                        # Create noisy input and target
                        noisy_x1 = x1 + sigma_a_sample * n
                        target = -n  # Target for AR-DAE is negative noise
                        
                        # Get model prediction
                        if self.args.model_time_conditioning == False:
                            output = sigma_a * self.net(noisy_x1)  # AR-DAE
                        else:
                            noise_amount = torch.zeros(x1.shape[0]).to(self.device) + sigma_a
                            output = sigma_a * self.net(x1=noisy_x1, time=noise_amount)  # AR-DAE with time conditioning
                        
                        # Calculate AR-DAE loss
                        loss_noise2score_loss = self.criterion_mse(output, target)
                    
                    # Apply scaling factor
                    loss_noise2score_loss *= self.args.lambda_noise2score_loss
                    
                    # Add to total loss
                    loss += loss_noise2score_loss
                    
                    # Logging
                    if self.args.DSM:
                        log_info += 'DSM noise_level:%f ' % (noise_amount.mean().item())
                    else:
                        log_info += 'AR-DAE sigma_a:%f ' % (sigma_a)
                    log_info += 'noise2score_loss:%.06f ' % (loss_noise2score_loss.item())
                    log_info += 'loss_sum:%f ' % (loss.mean().item())


                loss.backward()
                # total_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0, norm_type=2.0)
                # gradient clipping should not used with weight decay!
                clip_grad_norm_(self.net.parameters(), max_norm=1.0)  # Adjust max_norm as needed
                self.optimizer_G.step()
                if self.use_ema:
                    self.state['ema'].update(self.net.parameters())


                ## print information
                if j % self.args.log_freq == 0:
                    t1 = time.time()
                    # log_info += 'aug:%s ' % str(self.augmentation)
                    log_info += '%4.6fs/batch' % ((t1-t0)/self.args.log_freq)
                    if self.args.rank <= 0:
                        logging.info(log_info)
                    t0 = time.time()

                ## visualization
                if j % self.args.vis_freq == 0:
                    self.args.phase = 'eval'
                    self.net.eval()
                    if self.args.loss_r2r:
                        alpha = self.alpha_r2r
                        out_val = None
                        aver_num = 50
                        imgn_val = batch_samples['images']
                        eps = self.eps_r2r
                        for val_j in range(aver_num):
                            imgn_val_pert = imgn_val + alpha*eps*torch.FloatTensor(imgn_val.size()).normal_(mean=0, std=1.).cuda()
                            with torch.no_grad():
                                out_val_single = self.net(imgn_val_pert)
                            if out_val  is None:
                                out_val= out_val_single.detach()
                            else:
                                out_val += out_val_single.detach()
                            del out_val_single
                        output = torch.clamp(out_val/aver_num, 0., 1.)
                        vis_temps = [batch_samples['images'], output, batch_samples['plabels']] # for R2R dataset
                    elif self.args.loss_noise2score:
                        net_input = self.normalize(batch_samples['images'])
                        x1 = net_input
                        variance = self.args.noise_std ** 2 # needed for inference
                        if self.args.model_time_conditioning == False:
                            output = variance * self.net(x1)
                        else:
                            noise_amount = torch.zeros(x1.shape[0]).to(self.device)
                            output = variance * self.net(x1 = x1, time=noise_amount)
                        output = output + x1 
                        output = torch.clamp(output, 0., 1.)
                        output = self.unnormalize(output)
                        vis_temps = [batch_samples['images'],output, batch_samples['plabels']] # for N2S dataset -- net_input is masked 
                    elif self.args.loss_mdae:
                        # Build MDAE doubly-corrupted input for visualization
                        vis_labels = batch_samples['labels'] if 'labels' in batch_samples else batch_samples['plabels']
                        x1_vis = self.normalize(vis_labels)
                        B, C, H, W = x1_vis.shape

                        mask_pct = random.uniform(self.args.mdae_mask_pct_min, self.args.mdae_mask_pct_max)
                        spatial_mask = create_block_mask_2d(
                            B, H, W, mask_pct, block_size=self.args.mdae_block_size
                        ).to(self.device)
                        masked_input = x1_vis * spatial_mask
                        noise_amount = torch.rand(B).to(self.device)
                        corrupted_input = corrupt(masked_input, noise_amount, sde=self.args.sde,
                                                  max_sigma=self.args.ve_sde_sigma_max)
                        network_input = corrupted_input * spatial_mask
                        # Unnormalize for display
                        vis_mdae_input = torch.clamp(self.unnormalize(network_input), 0., 1.)

                        # Run model in clean-inference mode (time=0)
                        net_input = self.normalize(batch_samples['images'])
                        if self.args.model_time_conditioning:
                            noise_zero = torch.zeros(net_input.shape[0]).to(self.device)
                            net_output = self.net(x1=net_input, time=noise_zero)
                        else:
                            net_output = self.net(net_input)

                        if self.args.auto_normalize:
                            net_output = torch.clamp(net_output, -1., 1.)
                        else:
                            net_output = torch.clamp(net_output, 0., 1.)
                        unnormalized_output = self.unnormalize(net_output)

                        vis_temps = [vis_mdae_input, batch_samples['images'], unnormalized_output, vis_labels]

                    elif self.args.loss_c2s_sc_model or self.args.loss_d2s_sft or self.args.loss_d2s_pretrain or self.args.loss_composite:
                        net_input = self.normalize(batch_samples['images'])
                        if self.args.model_time_conditioning == False:
                            net_output = self.net(net_input)
                        else:
                            x1 = net_input
                            # todo -- needs to be fixed for model based approaches, not zero
                            noise_amount = torch.zeros(x1.shape[0]).to(self.device)
                            net_output = self.net(x1 = x1, time = noise_amount)

                        # clamp the output between -1 and 1 if self.args.auto_normalize is True, else clamp to 0 and 1
                        if self.args.auto_normalize:
                            net_output = torch.clamp(net_output, -1., 1.)
                        else:
                            net_output = torch.clamp(net_output, 0., 1.)

                        unnormalized_output = self.unnormalize(net_output)
                        # Handle both 'labels' and 'plabels' keys
                        vis_labels = batch_samples['labels'] if 'labels' in batch_samples else batch_samples['plabels']
                        vis_temps = [batch_samples['images'], unnormalized_output, vis_labels]
                    else:
                        vis_temps = [batch_samples['images'],self.net(batch_samples['images']), batch_samples['plabels']] # for N2S dataset -- net_input is masked 
                    self.net.train()
                    self.args.phase = 'train'
                    self.vis_results(i, j, vis_temps)
 
                steps += 1

            ## save networks (model only — skip optimizer/scheduler to speed up saves)
            if i % self.args.save_epoch_freq == 0:
                if self.args.rank <= 0:
                    logging.info('Saving state, epoch: %d iter:%d' % (i, 0))
                    self.save_networks('net', i)

            # start evaluation after ... epochs
            if i > self.args.eval_starting_epoch and  i % self.args.test_freq == 0:
                self.args.phase = 'eval'

                if self.use_ema:
                    ema = self.state['ema']
                    ema.store(self.net.parameters())
                    ema.copy_to(self.net.parameters())

                val_metrics = self.evaluate(epoch=i)
                psnr = val_metrics['psnr']
                ssim = val_metrics['ssim']
                lpips_val = val_metrics['lpips']
                proxy = val_metrics['proxy']
                psnr_std = val_metrics['psnr_std']
                ssim_std = val_metrics['ssim_std']
                logging.info('Mean: psnr:%.06f   ssim:%.06f   lpips:%.06f   proxy:%.06f' % (psnr, ssim, lpips_val, proxy))
                logging.info('Std : psnr:%.06f   ssim:%.06f ' % (psnr_std,ssim_std))
                if self.args.use_tb_logger:
                    if self.args.rank <= 0:
                        tb_logger.add_scalar('psnr_val_mean', psnr, i)
                        tb_logger.add_scalar('psnr_val_std', psnr_std, i)
                        tb_logger.add_scalar('ssim_val_mean', ssim, i)
                        tb_logger.add_scalar('ssim_val_std', ssim_std, i)
                        tb_logger.add_scalar('lpips_val_mean', lpips_val, i)
                        tb_logger.add_scalar('proxy_val_mean', proxy, i)

                current_metric = val_metrics.get(self.best_metric_name, psnr)
                is_best = current_metric > self.best_metric_value
                self.save_validation_metrics(i, val_metrics, is_best=is_best)

                if is_best:
                    self.best_metric_value = current_metric
                    if self.args.rank <= 0:
                        logging.info('best_%s:%.06f ' % (self.best_metric_name, self.best_metric_value))
                        logging.info('Saving state, epoch: %d iter:%d' % (i, 0))
                        self.save_networks('net', f'best_{self.best_metric_name}')
                        self.save_networks('optimizer_G', f'best_{self.best_metric_name}')
                        self.save_networks('scheduler', f'best_{self.best_metric_name}')
                    ## start data augmentation
                    if i > 30:
                        self.augmentation = self.args.data_augmentation
                if self.use_ema:
                    ema.restore(self.net.parameters())

                self.args.phase = 'train'

        ## end of training
        if self.args.rank <= 0:
            if self.args.use_tb_logger:
                tb_logger.close()
            self.save_networks('net', 'final')
            logging.info('The training stage on %s is over!!!' % (self.args.dataset))

        if self.args.dist:
            dist.destroy_process_group()


    def evaluate(self, epoch=None):
        """
        Memory-optimized evaluation method with MEF-aware time conditioning.
        Returns PSNR, SSIM, LPIPS, and a proxy challenge score.
        """
        self.net.eval()
        logging.info('Starting memory-optimized evaluation...')
        
        # Use lists to collect metrics instead of accumulating tensors
        psnr_values = []
        ssim_values = []
        lpips_values = []
        
        # Get patch size and stride from args, or use defaults
        patch_size = self.args.patch_size if hasattr(self.args, 'patch_size') else 256
        stride = self.args.stride if hasattr(self.args, 'stride') else 128
        
        # Use a smaller evaluation batch size to reduce memory usage
        eval_batch_size = min(2, self.args.batch_size) if hasattr(self.args, 'batch_size') else 1
        
        # Skip metric calculation if in challenge mode
        challenge_mode = hasattr(self.args, 'challenge') and self.args.challenge
        if challenge_mode:
            logging.info('Skipping evaluation metrics calculation in challenge mode')
            return {
                'epoch': epoch,
                'psnr': 0.0,
                'ssim': 0.0,
                'lpips': 0.0,
                'proxy': 0.0,
                'psnr_std': 0.0,
                'ssim_std': 0.0,
                'num_images': 0,
            }
        
        # Process with careful memory management
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(self.test_dataloader):
                # Process samples in smaller chunks to manage memory
                batch_samples = self.prepare(batch_samples)
                images = batch_samples['images']
                labels = batch_samples['labels']
                batch_size = images.shape[0]
                
                # Process images one by one to minimize memory usage
                for idx in range(0, batch_size, eval_batch_size):
                    # Clear GPU cache to free memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Extract sub-batch
                    end_idx = min(idx + eval_batch_size, batch_size)
                    sub_images = images[idx:end_idx]
                    sub_labels = labels[idx:end_idx]
                    
                    # Get number of channels from input images
                    num_channels = sub_images.shape[1]

                    # Apply normalization
                    sub_images_normalized = self.normalize(sub_images)
                    
                    # Determine noise level for time conditioning
                    sub_batch_size = sub_images.shape[0]
                    noise_amount = self.estimate_conditioning_sigma(sub_images_normalized)
                    
                    # Determine if we need patch-based processing
                    use_patch_based = (hasattr(self.args, 'enable_full_image') and 
                                    self.args.enable_full_image and 
                                    (sub_images.shape[2] > patch_size or sub_images.shape[3] > patch_size))
                    
                    # Process the image with appropriate method
                    if use_patch_based:
                        # Process with patch-based approach (memory efficient)
                        output = patch_based_restoration(
                            sub_images_normalized, 
                            model=self.net, 
                            r=stride, 
                            patch_size=patch_size,
                            time=noise_amount if self.args.model_time_conditioning else None,
                            model_time_conditioning=self.args.model_time_conditioning
                        )
                    else:
                        # Process directly based on model type
                        if self.args.model_time_conditioning:
                            if self.args.loss_noise2score:
                                sigma_0 = 2 * self.args.noise_std if self.args.auto_normalize else self.args.noise_std
                                variance = sigma_0 ** 2
                                output = variance * self.net(x1=sub_images_normalized, time=noise_amount)
                                output = output + sub_images_normalized
                            else:
                                output = self.net(x1=sub_images_normalized, time=noise_amount)
                        else:
                            if self.args.loss_noise2score:
                                sigma_0 = 2 * self.args.noise_std if self.args.auto_normalize else self.args.noise_std
                                variance = sigma_0 ** 2
                                output = variance * self.net(sub_images_normalized)
                                output = output + sub_images_normalized
                            elif self.args.loss_r2r:
                                # Modified R2R handling for lower memory usage
                                alpha = self.args.alpha_r2r
                                out_val = None
                                aver_num = 20  # Reduced from 50 for memory efficiency
                                eps = self.args.eps_r2r
                                
                                for val_j in range(aver_num):
                                    # Generate noise on CPU then move to GPU
                                    noise_tensor = torch.FloatTensor(sub_images.size()).normal_(mean=0, std=1.).to(self.device)
                                    imgn_val_pert = sub_images + alpha * eps * noise_tensor
                                    
                                    # Free up memory
                                    del noise_tensor
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    
                                    with torch.no_grad():
                                        out_val_single = self.net(imgn_val_pert)
                                    
                                    # Accumulate results
                                    if out_val is None:
                                        out_val = out_val_single
                                    else:
                                        out_val += out_val_single
                                    
                                    # Free memory
                                    del out_val_single, imgn_val_pert
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                
                                output = torch.clamp(out_val / aver_num, 0., 1.)
                                del out_val
                            else:
                                output = self.net(sub_images_normalized)
                    
                    # Clamp and unnormalize output
                    if self.args.auto_normalize:
                        output = torch.clamp(output, -1., 1.)
                    else:
                        output = torch.clamp(output, 0., 1.)
                    
                    output = self.unnormalize(output)
                    output = torch.clip(output, 0, 1)
                    
                    # Calculate metrics one image at a time to manage memory
                    for i in range(sub_batch_size):
                        output_img = output[i].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
                        gt_img = sub_labels[i].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
                        
                        # Calculate metrics
                        psnr = peak_signal_noise_ratio(output_img, gt_img, data_range=1)
                        ssim = structural_similarity(output_img, gt_img, data_range=1, channel_axis=2)
                        
                        psnr_values.append(psnr)
                        ssim_values.append(ssim)

                        if self.eval_lpips:
                            lpips_in = output[i:i+1] * 2 - 1
                            lpips_tgt = sub_labels[i:i+1] * 2 - 1
                            lpips_score = self.criterion_lpips(lpips_in, lpips_tgt).mean().item()
                            lpips_values.append(lpips_score)
                    
                    # Free memory
                    del sub_images, sub_labels, sub_images_normalized, output, noise_amount
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Log progress
                if batch_idx % 5 == 0:
                    logging.info(f"Processed {batch_idx} batches ({len(psnr_values)} images)")
        
        # Calculate and return metrics
        if len(psnr_values) > 0:
            psnr_mean = np.mean(psnr_values)
            ssim_mean = np.mean(ssim_values)
            psnr_std = np.std(psnr_values)
            ssim_std = np.std(ssim_values)
            lpips_mean = float(np.mean(lpips_values)) if len(lpips_values) > 0 else 0.0
        else:
            psnr_mean = ssim_mean = psnr_std = ssim_std = lpips_mean = 0.0

        proxy_score = self.compute_proxy_score(psnr_mean, ssim_mean, lpips_mean) if len(psnr_values) > 0 else 0.0
        
        logging.info(
            f'Evaluation metrics - PSNR: {psnr_mean:.4f} ± {psnr_std:.4f}, '
            f'SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}, '
            f'LPIPS: {lpips_mean:.4f}, Proxy: {proxy_score:.4f}'
        )
        
        return {
            'epoch': epoch,
            'psnr': float(psnr_mean),
            'ssim': float(ssim_mean),
            'lpips': float(lpips_mean),
            'proxy': float(proxy_score),
            'psnr_std': float(psnr_std),
            'ssim_std': float(ssim_std),
            'num_images': int(len(psnr_values)),
        }
    # def evaluate(self):
    #     """
    #     Evaluation method with support for full-image evaluation using patch-based denoising.
    #     Returns PSNR and SSIM metrics.
    #     """
    #     self.net.eval()
    #     logging.info('start testing (eval)...')
    #     logging.info('%d testing samples' % (self.test_dataset.__len__()))

    #     PSNR = []
    #     SSIM = []
        
    #     # Get patch size and stride from args, or use defaults
    #     patch_size = self.args.target_size if hasattr(self.args, 'patch_size') else 256
    #     stride = patch_size if hasattr(self.args, 'stride') else 128
        
    #     # Skip metric calculation if in challenge mode
    #     challenge_mode = hasattr(self.args, 'challenge') and self.args.challenge
    #     if challenge_mode:
    #         logging.info('Skipping evaluation metrics calculation in challenge mode')
    #         return 0.0, 0.0, 0.0, 0.0

    #     with torch.no_grad():
    #         for batch, batch_samples in enumerate(self.test_dataloader):
    #             batch_samples = self.prepare(batch_samples)
    #             images = batch_samples['images']
    #             labels = batch_samples['labels']
                
    #             # Get number of channels from input images
    #             num_channels = images.shape[1]
    #             is_color = (num_channels == 3)
                
    #             # Apply normalization
    #             images_normalized = self.normalize(images)
                
    #             # Determine if we need patch-based processing
    #             use_patch_based = (hasattr(self.args, 'enable_full_image') and 
    #                             self.args.enable_full_image and 
    #                             (images.shape[2] > patch_size or images.shape[3] > patch_size))
                
    #             # Determine noise level for time conditioning
    #             noise_amount = torch.zeros(images.shape[0]).to(self.device)
                
    #             if self.args.noise_model:
    #                 if self.args.auto_normalize:
    #                     sigma_0 = 2 * self.args.noise_std
    #                 else: 
    #                     sigma_0 = self.args.noise_std
    #                 noise_amount = torch.ones(images.shape[0]).to(self.device) * sigma_0
                
    #             if self.args.estimate_sigma:
    #                 sigma_values = torch.zeros(images.shape[0]).to(self.device)
    #                 for i in range(images.shape[0]):
    #                     if is_color:
    #                         # For color images, estimate sigma for each channel separately and average
    #                         channel_sigmas = []
    #                         for c in range(num_channels):
    #                             img_channel = images_normalized[i, c].detach().cpu().numpy()
    #                             channel_sigmas.append(estimate_sigma(img_channel))
    #                         sigma_values[i] = torch.tensor(np.mean(channel_sigmas), device=self.device)
    #                     else:
    #                         # For grayscale, estimate sigma on the single channel
    #                         img = images_normalized[i, 0].detach().cpu().numpy()
    #                         sigma_values[i] = torch.tensor(estimate_sigma(img), device=self.device)
                    
    #                 noise_amount = sigma_values
                
    #             # Process the image with or without patch-based approach
    #             if use_patch_based:
    #                 # Patch-based processing for large images
    #                 output = patch_based_restoration(
    #                     images, 
    #                     model=self.net, 
    #                     r=stride, 
    #                     patch_size=patch_size,
    #                     time=noise_amount if self.args.model_time_conditioning else None,
    #                     model_time_conditioning=self.args.model_time_conditioning
    #                 )
    #             else:
    #                 # Direct processing for smaller images
    #                 if self.args.model_time_conditioning:
    #                     if self.args.loss_noise2score:
    #                         if self.args.auto_normalize:
    #                             sigma_0 = 2 * self.args.noise_std
    #                         else: 
    #                             sigma_0 = self.args.noise_std
    #                         variance = sigma_0 ** 2 # needed for inference
    #                         output = variance * self.net(x1=images_normalized, time=noise_amount)  
    #                         output = output + images_normalized
    #                     else:
    #                         # Regular time-conditioned model inference
    #                         output = self.net(x1=images_normalized, time=noise_amount)
    #                 else:
    #                     # Without time conditioning
    #                     if self.args.loss_noise2score:
    #                         if self.args.auto_normalize:
    #                             sigma_0 = 2 * self.args.noise_std
    #                         else: 
    #                             sigma_0 = self.args.noise_std
    #                         variance = sigma_0 ** 2
    #                         output = variance * self.net(images_normalized)
    #                         output = output + images_normalized
    #                     elif self.args.loss_r2r:
    #                         # Special handling for R2R loss
    #                         alpha = self.args.alpha_r2r
    #                         out_val = None
    #                         aver_num = 50
    #                         imgn_val = batch_samples['images']
    #                         eps = self.args.eps_r2r
    #                         for val_j in range(aver_num):
    #                             imgn_val_pert = imgn_val + alpha*eps*torch.FloatTensor(imgn_val.size()).normal_(mean=0, std=1.).to(self.device)
    #                             with torch.no_grad():
    #                                 out_val_single = self.net(imgn_val_pert)
    #                             if out_val is None:
    #                                 out_val = out_val_single.detach()
    #                             else:
    #                                 out_val += out_val_single.detach()
    #                             del out_val_single
    #                         output = torch.clamp(out_val/aver_num, 0., 1.)
    #                     else:
    #                         # Standard non-time-conditioned inference
    #                         output = self.net(images_normalized)
                
    #             # Clamp and unnormalize output
    #             if self.args.auto_normalize:
    #                 output = torch.clamp(output, -1., 1.)
    #             else:
    #                 output = torch.clamp(output, 0., 1.)
    #             output = self.unnormalize(output)
    #             output = torch.clip(output, 0, 1)
                
    #             # Process each sample in the batch
    #             for idx in range(output.shape[0]):
    #                 # Convert tensors to numpy for evaluation
    #                 if is_color:
    #                     # For color images, convert to proper format for PSNR/SSIM
    #                     output_img = output[idx].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    #                     gt_img = labels[idx].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
                        
    #                     # Calculate metrics for color images
    #                     psnr = peak_signal_noise_ratio(output_img, gt_img, data_range=1)
    #                     ssim = structural_similarity(output_img, gt_img, data_range=1, channel_axis=2)
    #                 else:
    #                     # For grayscale images, use original approach
    #                     output_img = output[idx, 0].detach().cpu().numpy().astype(np.float32)
    #                     gt_img = labels[idx, 0].detach().cpu().numpy().astype(np.float32)
                        
    #                     psnr = peak_signal_noise_ratio(output_img, gt_img, data_range=1)
    #                     ssim = structural_similarity(output_img, gt_img, data_range=1)
                    
    #                 PSNR.append(psnr)
    #                 SSIM.append(ssim)
                    
    #                 # Log individual sample metrics for debugging
    #                 if use_patch_based and batch % 20 == 0:  # Log every 5th batch to avoid too much output
    #                     img_name = batch_samples['name'][idx] if 'name' in batch_samples else f"sample_{batch}_{idx}"
    #                     logging.info(f'Eval image {img_name} - PSNR: {psnr:.4f}, SSIM: {ssim:.4f}')

    #     # Calculate and return metrics
    #     psnr_mean = np.mean(PSNR)
    #     ssim_mean = np.mean(SSIM)
    #     psnr_std = np.std(PSNR)
    #     ssim_std = np.std(SSIM)
        
    #     logging.info(f'Evaluation metrics - PSNR: {psnr_mean:.4f} ± {psnr_std:.4f}, SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}')

    #     return psnr_mean, ssim_mean, psnr_std, ssim_std


    def geometric_ensemble_inference(self, image, noise_amount=None, patch_size=None, stride=None, full_mode=False):
        """
        Apply geometric self-ensemble technique to image denoising with adaptive patch size support.
        
        Args:
            image: Input noisy image tensor [B, C, H, W]
            noise_amount: Noise level for time conditioning (optional)
            patch_size: Size of patch to use for patch-based processing
            stride: Stride to use for overlapping patches
            full_mode: Whether to use adaptive patch sizes
            
        Returns:
            Ensemble denoised output tensor [B, C, H, W]
        """
        # Ensure we're working with a batch of size 1 for simplicity
        assert image.size(0) == 1, "Geometric ensemble expects batch size 1"
        
        model = self.net  # Use the class model
        B, C, H, W = image.size()
        
        # Define the 8 geometric transformations
        # 1: Identity (original), 2: Horizontal flip, 3: Vertical flip, 4: 90° rotation
        # 5: 180° rotation, 6: 270° rotation, 7: Horizontal+90°, 8: Vertical+90°
        
        # Initialize storage for outputs
        ensemble_outputs = []
        
        # If there's no time conditioning argument but it's requested, create a default
        if self.args.model_time_conditioning and noise_amount is None:
            noise_amount = torch.zeros(1).to(self.device)
        
        # Loop through all 8 transformations
        for t in range(8):
            # Apply the geometric transformation to the input
            if t == 0:  # Identity
                transformed_img = image
            elif t == 1:  # Horizontal flip
                transformed_img = torch.flip(image, dims=[3])
            elif t == 2:  # Vertical flip
                transformed_img = torch.flip(image, dims=[2])
            elif t == 3:  # 90° rotation
                transformed_img = torch.rot90(image, k=1, dims=[2, 3])
            elif t == 4:  # 180° rotation
                transformed_img = torch.rot90(image, k=2, dims=[2, 3])
            elif t == 5:  # 270° rotation
                transformed_img = torch.rot90(image, k=3, dims=[2, 3])
            elif t == 6:  # Horizontal flip + 90° rotation
                transformed_img = torch.rot90(torch.flip(image, dims=[3]), k=1, dims=[2, 3])
            elif t == 7:  # Vertical flip + 90° rotation
                transformed_img = torch.rot90(torch.flip(image, dims=[2]), k=1, dims=[2, 3])
            
            # Get dimensions of transformed image (important for rotations)
            t_B, t_C, t_H, t_W = transformed_img.size()
            
            # Determine if we need patch-based processing for this transformed image
            use_patch_based = (full_mode and (t_H > patch_size or t_W > patch_size))
            
            # Process the image with or without patches
            if use_patch_based:
                # For patch-based with ensemble, process each patch
                h, w = t_H, t_W
                
                # Calculate needed padding
                H = (h - 1) // stride * stride + patch_size
                W = (w - 1) // stride * stride + patch_size
                padh = H - h if h % patch_size != 0 else 0
                padw = W - w if w % patch_size != 0 else 0
                
                if padh != 0 or padw != 0:
                    transformed_img = F.pad(transformed_img, (0, padw, 0, padh), 'reflect')
                
                # Create a tensor to store the outputs
                patch_output = torch.zeros_like(transformed_img)
                count = torch.zeros_like(transformed_img)
                
                # Process each patch 
                for i in range(0, H - patch_size + 1, stride):
                    for j in range(0, W - patch_size + 1, stride):
                        patch = transformed_img[:, :, i:i+patch_size, j:j+patch_size]
                        
                        # Process this patch
                        with torch.no_grad():
                            if self.args.model_time_conditioning:
                                patch_result = model(x1=patch, time=noise_amount)
                            else:
                                if hasattr(self.args, 'loss_noise2score') and self.args.loss_noise2score:
                                    variance = self.args.noise_std ** 2
                                    patch_result = variance * model(patch)
                                    patch_result = patch_result + patch
                                elif hasattr(self.args, 'loss_r2r') and self.args.loss_r2r:
                                    # Handle R2R loss case
                                    alpha = self.args.alpha_r2r
                                    out_val = None
                                    aver_num = 10  # Reduced for speed
                                    eps = self.args.eps_r2r
                                    for val_j in range(aver_num):
                                        img_pert = patch + alpha*eps*torch.FloatTensor(patch.size()).normal_(mean=0, std=1.).to(self.device)
                                        with torch.no_grad():
                                            out_val_single = model(img_pert)
                                        if out_val is None:
                                            out_val = out_val_single.detach()
                                        else:
                                            out_val += out_val_single.detach()
                                        del out_val_single
                                    patch_result = torch.clamp(out_val/aver_num, 0., 1.)
                                else:
                                    patch_result = model(patch)
                        
                        # Add result to output accumulator
                        patch_output[:, :, i:i+patch_size, j:j+patch_size] += patch_result
                        count[:, :, i:i+patch_size, j:j+patch_size] += 1
                
                # Average overlapping regions
                transformed_output = patch_output / count
                # Crop to original size (remove padding)
                transformed_output = transformed_output[:, :, :h, :w]
            else:
                # Direct processing without patches
                with torch.no_grad():
                    if self.args.model_time_conditioning:
                        transformed_output = model(x1=transformed_img, time=noise_amount)
                    else:
                        if hasattr(self.args, 'loss_noise2score') and self.args.loss_noise2score:
                            variance = self.args.noise_std ** 2
                            transformed_output = variance * model(transformed_img)
                            transformed_output = transformed_output + transformed_img
                        elif hasattr(self.args, 'loss_r2r') and self.args.loss_r2r:
                            # Handle R2R loss case
                            alpha = self.args.alpha_r2r
                            out_val = None
                            aver_num = 10  # Reduced for speed
                            eps = self.args.eps_r2r
                            for val_j in range(aver_num):
                                img_pert = transformed_img + alpha*eps*torch.FloatTensor(transformed_img.size()).normal_(mean=0, std=1.).to(self.device)
                                with torch.no_grad():
                                    out_val_single = model(img_pert)
                                if out_val is None:
                                    out_val = out_val_single.detach()
                                else:
                                    out_val += out_val_single.detach()
                                del out_val_single
                            transformed_output = torch.clamp(out_val/aver_num, 0., 1.)
                        else:
                            transformed_output = model(transformed_img)
            
            # Apply the inverse transformation to align with original geometry
            if t == 0:  # Identity - no change needed
                aligned_output = transformed_output
            elif t == 1:  # Horizontal flip - flip back horizontally
                aligned_output = torch.flip(transformed_output, dims=[3])
            elif t == 2:  # Vertical flip - flip back vertically
                aligned_output = torch.flip(transformed_output, dims=[2])
            elif t == 3:  # 90° rotation - rotate back 270°
                aligned_output = torch.rot90(transformed_output, k=3, dims=[2, 3])
            elif t == 4:  # 180° rotation - rotate back 180°
                aligned_output = torch.rot90(transformed_output, k=2, dims=[2, 3])
            elif t == 5:  # 270° rotation - rotate back 90°
                aligned_output = torch.rot90(transformed_output, k=1, dims=[2, 3])
            elif t == 6:  # Horizontal flip + 90° rotation - rotate back 270° then flip horizontally
                aligned_output = torch.flip(torch.rot90(transformed_output, k=3, dims=[2, 3]), dims=[3])
            elif t == 7:  # Vertical flip + 90° rotation - rotate back 270° then flip vertically
                aligned_output = torch.flip(torch.rot90(transformed_output, k=3, dims=[2, 3]), dims=[2])
            
            # Add to ensemble collection
            ensemble_outputs.append(aligned_output)
        
        # Average all outputs
        ensemble_result = torch.mean(torch.stack(ensemble_outputs, dim=0), dim=0)
        
        return ensemble_result


    def test(self):
        """
        Enhanced test method with geometric self-ensemble augmentation and adaptive patch sizes.
        """
        self.net.eval()
        logging.info('Starting testing with geometric self-ensemble and adaptive patch sizes...')

        PSNR = []
        SSIM = []
        total_processing_time = 0
        total_images = 0

        logging.info(f'Found {self.test_dataset.__len__()} testing samples')
        logging.info(f'Using batch size: 1 (required for geometric ensemble)')
        
        # Define adaptive patch sizes
        adaptive_patch_sizes = {
            'large': 896,
            'medium': 768,
            'small': 512
        }
        
        # Get default patch size and stride (fallback if not using adaptive mode)
        default_patch_size = self.args.patch_size if hasattr(self.args, 'patch_size') else 256
        default_stride = self.args.stride if hasattr(self.args, 'stride') else 128
        
        # Check if full/adaptive resolution mode is enabled
        full_mode = (
            hasattr(self.args, 'full') and self.args.full or 
            hasattr(self.args, 'enable_full_image') and self.args.enable_full_image
        )
        
        if full_mode:
            logging.info("ADAPTIVE RESOLUTION MODE ENABLED")
            logging.info(f"Available patch sizes: {adaptive_patch_sizes['large']}x{adaptive_patch_sizes['large']} (large), "
                        f"{adaptive_patch_sizes['medium']}x{adaptive_patch_sizes['medium']} (medium), "
                        f"{adaptive_patch_sizes['small']}x{adaptive_patch_sizes['small']} (small)")
        else:
            logging.info(f"Using fixed patch size: {default_patch_size}x{default_patch_size}")
            logging.info(f"Using fixed stride: {default_stride}")
        
        # Determine if we're in challenge mode
        challenge_mode = hasattr(self.args, 'challenge') and self.args.challenge
        if challenge_mode:
            logging.info('Running in challenge mode - no ground truth metrics will be calculated!!!')
        
        # Create output directory for results
        save_dir = os.path.join(self.args.save_folder, 'results_ensemble')
        os.makedirs(save_dir, exist_ok=True)

        # Need to use batch size 1 for geometric ensemble
        original_batch_size = self.args.batch_size
        self.args.batch_size = 1
        
        # Create a new dataloader with batch size 1
        ensemble_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(ensemble_dataloader):
                start_time = time.time()
                
                # Prepare batch data
                batch_samples = self.prepare(batch_samples)
                images = batch_samples['images']
                
                # Get image names for saving
                img_names = batch_samples['name'] if 'name' in batch_samples else [f"img_{total_images}.png"]
                
                # Get number of channels from input images
                num_channels = images.shape[1]
                is_color = (num_channels == 3)
                
                # Normalize input images
                images_normalized = self.normalize(images)
                
                # Apply time conditioning
                noise_amount = torch.zeros(1).to(self.device)
                
                if self.args.noise_model:
                    # Apply fixed noise level
                    if self.args.auto_normalize:
                        sigma_0 = 2 * self.args.noise_std
                    else:
                        sigma_0 = self.args.noise_std
                    noise_amount = torch.ones(1).to(self.device) * sigma_0
                    logging.info(f"Using fixed noise level: {sigma_0}")
                
                if self.args.estimate_sigma:
                    # Estimate noise level from image
                    if is_color:
                        channel_sigmas = []
                        for c in range(num_channels):
                            img_channel = images_normalized[0, c].detach().cpu().numpy()
                            channel_sigmas.append(estimate_sigma(img_channel))
                        noise_amount[0] = torch.tensor(np.mean(channel_sigmas), device=self.device)
                    else:
                        img = images_normalized[0, 0].detach().cpu().numpy()
                        noise_amount[0] = torch.tensor(estimate_sigma(img), device=self.device)
                
                # Select appropriate patch size and stride based on image dimensions
                img_height, img_width = images.shape[2], images.shape[3]
                
                if full_mode:
                    if img_height >= adaptive_patch_sizes['large'] and img_width >= adaptive_patch_sizes['large']:
                        patch_size = adaptive_patch_sizes['large']
                        category = "large"
                    elif img_height >= adaptive_patch_sizes['medium'] and img_width >= adaptive_patch_sizes['medium']:
                        patch_size = adaptive_patch_sizes['medium']
                        category = "medium"
                    elif img_height >= adaptive_patch_sizes['small'] and img_width >= adaptive_patch_sizes['small']:
                        patch_size = adaptive_patch_sizes['small']
                        category = "small"
                    else:
                        # Image is too small for any of our patch sizes, don't use patch-based
                        patch_size = min(img_height, img_width)
                        category = "direct"
                    
                    # Set stride to half of patch size by default if not explicitly specified
                    stride = getattr(self.args, 'stride', patch_size // 2)
                    
                    img_name = img_names[0]
                    use_patch_based = (img_height > patch_size or img_width > patch_size)
                    logging.info(f"Processing image {total_images}: {img_name}, shape=({img_height}×{img_width}), "
                                f"category={category}, patch_size={patch_size}, stride={stride}")
                else:
                    # In non-adaptive mode, use the default patch size/stride
                    patch_size = default_patch_size
                    stride = default_stride
                    use_patch_based = (img_height > patch_size or img_width > patch_size)
                    logging.info(f"Processing image {total_images}: {img_names[0]}, shape=({img_height}×{img_width}), "
                                f"use_patch_based={use_patch_based}, patch_size={patch_size}")
                
                # Process image with geometric self-ensemble and adaptive patch size
                output = self.geometric_ensemble_inference(
                    image=images_normalized,
                    noise_amount=noise_amount if self.args.model_time_conditioning else None,
                    patch_size=patch_size,
                    stride=stride,
                    full_mode=full_mode
                )
                
                # Post-process outputs
                if self.args.auto_normalize:
                    output = torch.clamp(output, -1., 1.)
                else:
                    output = torch.clamp(output, 0., 1.)
                
                output = self.unnormalize(output)
                output = torch.clip(output, 0, 1)
                
                # Measure processing time
                batch_time = time.time() - start_time
                total_processing_time += batch_time
                total_images += 1
                
                logging.info(f"Image {batch_idx} processed in {batch_time:.2f}s (with adaptive patches and geometric ensemble)")
                
                # Save image
                if challenge_mode:
                    save_path = os.path.join(save_dir, img_names[0])
                    self.save_image_tensor(output[0], save_path)
                
                # Visualize
                if batch_idx % self.args.vis_freq_test == 0:
                    if not challenge_mode:
                        labels = batch_samples['labels']
                        vis_temps = [images, output, labels]
                        self.vis_test_results(batch_idx, vis_temps)
                
                # Calculate metrics if not in challenge mode
                if not challenge_mode:
                    labels = batch_samples['labels']
                    
                    if is_color:
                        # For color images
                        output_img = output[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
                        gt_img = labels[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
                        
                        # Calculate metrics
                        psnr = peak_signal_noise_ratio(output_img, gt_img, data_range=1)
                        ssim = structural_similarity(output_img, gt_img, data_range=1, channel_axis=2)
                    else:
                        # For grayscale images
                        output_img = output[0, 0].detach().cpu().numpy().astype(np.float32)
                        gt_img = labels[0, 0].detach().cpu().numpy().astype(np.float32)
                        
                        psnr = peak_signal_noise_ratio(output_img, gt_img, data_range=1)
                        ssim = structural_similarity(output_img, gt_img, data_range=1)
                    
                    PSNR.append(psnr)
                    SSIM.append(ssim)
                    
                    # Log individual metrics
                    logging.info(f'Image {img_names[0]} - PSNR: {psnr:.4f}, SSIM: {ssim:.4f}')
        
        # Restore original batch size
        self.args.batch_size = original_batch_size
        
        # Log processing statistics
        avg_time_per_image = total_processing_time / total_images
        logging.info(f'Total processing time: {total_processing_time:.2f}s for {total_images} images')
        logging.info(f'Average time per image: {avg_time_per_image:.4f}s')
        
        # Calculate and log average metrics
        if not challenge_mode and len(PSNR) > 0:
            psnr_mean = np.mean(PSNR)
            ssim_mean = np.mean(SSIM)
            psnr_std = np.std(PSNR)
            ssim_std = np.std(SSIM)
            
            logging.info('-------- ADAPTIVE GEOMETRIC ENSEMBLE TEST RESULTS --------')
            logging.info(f'Average PSNR: {psnr_mean:.4f} ± {psnr_std:.4f}')
            logging.info(f'Average SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}')
            
            # Save metrics to a text file
            metrics_path = os.path.join(self.args.save_folder, 'metrics_adaptive_ensemble.txt')
            with open(metrics_path, 'w') as f:
                f.write(f'Average PSNR: {psnr_mean:.4f} ± {psnr_std:.4f}\n')
                f.write(f'Average SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}\n')
                f.write(f'Total processing time: {total_processing_time:.2f}s\n')
                f.write(f'Average time per image: {avg_time_per_image:.4f}s\n')
                f.write(f'Geometric ensemble enabled: Yes (8 transformations)\n')
                
                if full_mode:
                    f.write(f'Adaptive patch sizes: {adaptive_patch_sizes["large"]}×{adaptive_patch_sizes["large"]}, '
                        f'{adaptive_patch_sizes["medium"]}×{adaptive_patch_sizes["medium"]}, '
                        f'{adaptive_patch_sizes["small"]}×{adaptive_patch_sizes["small"]}\n')
                else:
                    f.write(f'Fixed patch size: {default_patch_size}×{default_patch_size}\n')
                
                f.write('\nIndividual results:\n')
                for i, (psnr, ssim) in enumerate(zip(PSNR, SSIM)):
                    img_name = self.test_dataset.img_list[i].split('/')[-1] if hasattr(self.test_dataset, 'img_list') else f"img_{i}.png"
                    f.write(f'{img_name}: PSNR={psnr:.4f}, SSIM={ssim:.4f}\n')
        else:
            logging.info('-------- CHALLENGE MODE WITH ADAPTIVE GEOMETRIC ENSEMBLE --------')
            logging.info(f'Processed and saved {total_images} images to {save_dir}')
            
            # Save a summary file for challenge mode
            summary_path = os.path.join(self.args.save_folder, 'challenge_adaptive_ensemble_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f'Challenge mode test with adaptive geometric ensemble completed on {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write(f'Processed {total_images} images\n')
                f.write(f'Model: {self.args.resume}\n')
                f.write(f'Total processing time: {total_processing_time:.2f}s\n')
                f.write(f'Average time per image: {avg_time_per_image:.4f}s\n')
                f.write(f'Geometric ensemble enabled: Yes (8 transformations)\n')
                
                if full_mode:
                    f.write(f'Adaptive patch sizes: {adaptive_patch_sizes["large"]}×{adaptive_patch_sizes["large"]}, '
                        f'{adaptive_patch_sizes["medium"]}×{adaptive_patch_sizes["medium"]}, '
                        f'{adaptive_patch_sizes["small"]}×{adaptive_patch_sizes["small"]}\n')
                else:
                    f.write(f'Fixed patch size: {default_patch_size}×{default_patch_size}\n')
                
                if self.args.model_time_conditioning:
                    if self.args.estimate_sigma:
                        f.write(f'Used estimated noise levels for time conditioning\n')
                    else:
                        f.write(f'Used fixed noise level: {self.args.noise_std} for time conditioning\n')
                else:
                    f.write(f'No time conditioning used\n')




    def save_image_tensor(self, tensor, path):
        """
        Save a tensor as an image file.
        
        Args:
            tensor: Image tensor of shape [C, H, W] with values in [0, 1]
            path: Path to save the image
        """
        if tensor.shape[0] == 3:  # RGB
            # Convert from [C, H, W] to [H, W, C] format
            img_np = tensor.permute(1, 2, 0).detach().cpu().numpy()
            # Convert to uint8
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            # Save using PIL
            Image.fromarray(img_np).save(path)
        else:  # Grayscale
            img_np = tensor[0].detach().cpu().numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_np, mode='L').save(path)

        def save_image(self, tensor, path):
            img = Image.fromarray(((tensor/2.0 + 0.5).data.cpu().numpy()*255).transpose((1, 2, 0)).astype(np.uint8))
            img.save(path)

    def load_networks(self, net_name, resume, strict=True):
        load_path = resume
        network = getattr(self, net_name)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path, map_location=torch.device(self.device))
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v

        # Adapt init_conv weights when in_channels differs (e.g. 3 → 15 for MEF)
        in_channels = getattr(self.args, 'in_channels', 3)
        if in_channels != 3 and 'init_conv.weight' in load_net_clean:
            old_w = load_net_clean['init_conv.weight']
            out_c, old_in, kH, kW = old_w.shape
            if old_in != in_channels:
                repeats = (in_channels // old_in) + 1
                new_w = old_w.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]
                new_w = new_w / repeats
                load_net_clean['init_conv.weight'] = new_w
                if self.args.rank <= 0:
                    logging.info(f'  Adapted init_conv weights: {old_in} → {in_channels} channels')

        # Init skip_proj to zeros if not in checkpoint
        if in_channels != 3 and 'skip_proj.weight' not in load_net_clean:
            if getattr(self.args, 'zero_skip_proj', False):
                load_net_clean['skip_proj.weight'] = torch.zeros(3, in_channels, 1, 1)
            else:
                # Xavier init
                load_net_clean['skip_proj.weight'] = torch.randn(3, in_channels, 1, 1) * 0.01
            if self.args.rank <= 0:
                logging.info(f'  Initialized skip_proj.weight for {in_channels} → 3 channels')

        if 'optimizer' or 'scheduler' in net_name:
            network.load_state_dict(load_net_clean)
        else:
            network.load_state_dict(load_net_clean, strict=strict)
        del load_net_clean

    def save_networks(self, net_name, epoch):
        network = getattr(self, net_name)
        save_filename = '{}_{}.pth'.format(net_name, epoch)
        save_path = os.path.join(self.args.snapshot_save_dir, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        # Always move tensors to CPU before saving (fixes slow GPU→disk writes on Lustre)
        for key, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

 
