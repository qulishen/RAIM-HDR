import os
import time
import logging
import math
import argparse
import numpy as np
import torch
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data as data
import warnings
from utils.util import setup_logger, print_args
from models import Trainer

def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='M4Raw && fastMRI Training')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--phase', default='train', type=str)
    parser.add_argument('--name', default='...', type=str)  # name to save


    ## device setting
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)

    ## network setting
    parser.add_argument('--net_name', default='TwoCon_Joint_Translate_Denoise_v1', type=str, help='RESTORMER | RESUNET | NAFNET')
    parser.add_argument('--input_nc', default=1, type=int)
    parser.add_argument('--output_nc', default=1, type=int)


    ## dataloader setting
    # parser.add_argument('--traindata_root', default='/srv/local/shared/data/M4Raw/M4RawV1.5_multicoil_train/multicoil_train',type=str)
    # parser.add_argument('--testdata_root', default='/srv/local/shared/data/M4Raw/M4RawV1.5_multicoil_val/multicoil_val',type=str)
    parser.add_argument('--dataset', default='M4Raw', type=str, help='M4Raw | M4Raw_N2N | M4Raw_N2S | fastMRI | fastMRI_N2N')
    # parser.add_argument('--modal', default='T1_T2_FLAIR', type=str, help='T1 | T2 | FLAIR | ALL | T1_T2 | T2_FLAIR | ...')
    parser.add_argument('--trainset', default=None, type=str, help='TrainSet | TrainSet_N2N | TrainSet_N2S | FastMRITrainSet')
    parser.add_argument('--testset', default='TestSet_T2_T1_N2N', type=str, help='TestSet | FastMRITestSet')
    parser.add_argument('--save_test_root', default='generated', type=str)
    parser.add_argument('--in_channels', default=3, type=int,
                        help='Number of input channels (e.g. 15 for 5-frame MEF RGB)')
    parser.add_argument('--zero_skip_proj', action='store_true',
                        help='Init skip_proj to zeros (for cross-domain tasks)')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--noise_level', default=13, type=int)
    parser.add_argument('--data_augmentation', action='store_true')

    ## optim setting
    parser.add_argument('--weight_decay', default=1e-3, type=float) # 1e-4
    parser.add_argument('--start_iter', default=0, type=int)
    parser.add_argument('--max_iter', default=500, type=int)
    parser.add_argument('--eval_starting_epoch', default=0, type=int)

    # need to turn on one of the losses
    parser.add_argument('--loss_l1', action='store_true')
    parser.add_argument('--loss_n2s', action='store_true')

    parser.add_argument('--loss_c2s_sc_model', action='store_true')
    parser.add_argument('--loss_d2s_sft', action='store_true')
    parser.add_argument('--loss_d2s_pretrain', action='store_true') # pretrain from clean images -- corrupted clean images -- self)
    # lambda_d2s_sft
    parser.add_argument('--lambda_d2s_sft', default=1, type=float)
    # lambda_d2s_pretrain
    parser.add_argument('--lambda_d2s_pretrain', default=1, type=float)

    # MDAE (Masked Denoising Autoencoder) loss
    parser.add_argument('--loss_mdae', action='store_true')
    parser.add_argument('--lambda_mdae', default=1, type=float, help='Overall MDAE loss weight')
    parser.add_argument('--lambda_mdae_masked', default=1, type=float, help='Weight for masked reconstruction loss')
    parser.add_argument('--lambda_mdae_visible', default=1, type=float, help='Weight for visible denoising loss')
    parser.add_argument('--mdae_mask_pct_min', default=0.01, type=float, help='Min masking percentage')
    parser.add_argument('--mdae_mask_pct_max', default=0.75, type=float, help='Max masking percentage')
    parser.add_argument('--mdae_block_size', default=16, type=int, help='Block size for MDAE masking')

    parser.add_argument('--loss_noiser2noise', action='store_true')
    # alpha_noiser2noise
    parser.add_argument('--alpha_noiser2noise', default=0.0005, type=float)

    parser.add_argument('--loss_r2r', action='store_true')

    parser.add_argument('--loss_mse', action='store_true')
    parser.add_argument('--loss_adv', action='store_true')
    parser.add_argument('--gan_type', default='WGAN_GP', type=str)
    parser.add_argument('--loss_noise2score', action='store_true', help='use noise2score loss')

    # Composite supervised loss (L1 + SSIM + LPIPS) — directly optimizes evaluation metric
    parser.add_argument('--loss_composite', action='store_true',
                        help='Composite L1+SSIM+LPIPS supervised loss for image restoration')
    parser.add_argument('--lambda_composite_l1', default=1.0, type=float, help='L1 weight in composite loss')
    parser.add_argument('--lambda_ssim', default=1.0, type=float, help='SSIM weight in composite loss')
    parser.add_argument('--lambda_lpips', default=0.1, type=float, help='LPIPS weight in composite loss')
    parser.add_argument('--lambda_grad', default=0.0, type=float, help='Gradient consistency weight in composite loss')
    parser.add_argument('--lambda_sat', default=0.0, type=float, help='Exposure-extreme masked reconstruction weight in composite loss')
    parser.add_argument('--sat_threshold_low', default=0.02, type=float, help='Lower exposure threshold for masked loss')
    parser.add_argument('--sat_threshold_high', default=0.98, type=float, help='Upper exposure threshold for masked loss')



    # Diffusion related: timesteps
    parser.add_argument('--model_time_conditioning', action='store_true', help='use time conditioning for the model')

    parser.add_argument('--timesteps', default=1000, type=int)
    parser.add_argument('--beta_schedule', default='linear', type=str)
    parser.add_argument('--sde', default='ve', type=str, help='ve | ddpm')
    parser.add_argument('--ve_sde_sigma_min', default=1e-1, type=float)
    parser.add_argument('--ve_sde_sigma_max', default=5, type=float)


    

    parser.add_argument('--net_3D', action='store_true') # for 3D dataset

    parser.add_argument('--lambda_l1', default=1, type=float)
    parser.add_argument('--lambda_r2r', default=1, type=float)
    parser.add_argument('--eps_r2r', default=0.025, type=float)
    parser.add_argument('--alpha_r2r', default=0.5, type=float)

    parser.add_argument('--lambda_n2s', default=1, type=float)
    parser.add_argument('--lambda_c2s_sc_model', default=1, type=float)
    parser.add_argument('--lambda_mse', default=1, type=float)
    parser.add_argument('--lambda_perceptual', default=1, type=float)
    parser.add_argument('--lambda_adv', default=5e-3, type=float)
    parser.add_argument('--lambda_noise2score_loss', default=1, type=float)

    # In your argument parser section
    parser.add_argument('--lambda_d2s_sft_mse', type=float, default=1.0, 
                        help='Weight for MSE component of D2S fine-tuning loss')
    parser.add_argument('--lambda_d2s_sft_l1', type=float, default=1, 
                        help='Weight for L1 component of D2S fine-tuning loss')
    parser.add_argument('--lambda_d2s_sft_sobel', type=float, default=0.5, 
                        help='Weight for Sobel (edge) component of D2S fine-tuning loss')

    parser.add_argument('--resume', default= '', type=str)
    parser.add_argument('--resume_optim', default='', type=str)
    parser.add_argument('--resume_scheduler', default='', type=str)


    ## normalization setting
    parser.add_argument('--auto_normalize', action='store_true')    # normalize range to [-1, 1], for gaussian corruption
    

    ## log setting
    parser.add_argument('--log_freq', default=10, type=int)
    parser.add_argument('--vis_freq', default=300000, type=int) #50000
    parser.add_argument('--vis_freq_test', default=10, type=int) #50000
    parser.add_argument('--save_epoch_freq', default=100, type=int) #100
    parser.add_argument('--test_freq', default=10, type=int) #100
    parser.add_argument('--save_folder', default='/home/jtu9/Desktop/LamLab_Share/Group/Jiachen_jtu9/logs_M4Raw/joint_denoise_2_2/T1T2', type=str)
    parser.add_argument('--vis_step_freq', default=100, type=int)
    parser.add_argument('--use_tb_logger', action='store_true')
    parser.add_argument('--save_test_results', action='store_true')
    parser.add_argument('--best_metric', default='psnr', choices=['proxy', 'psnr', 'ssim'],
                        help='Validation metric used to select the best checkpoint')
    parser.add_argument('--eval_lpips', action='store_true',
                        help='Compute LPIPS during validation and log a proxy challenge score')

    ## test / eval setting
    parser.add_argument('--noise_model', action='store_true') # if considering noise model
    parser.add_argument('--estimate_sigma', action='store_true') # if considering noise model, estimate sigma via skimage's package
    parser.add_argument('--noise_std', default=0.098, type=float) # input sigma mannually, default to 0.025 (std for the M4Raw dataset)
    

    ## monte carlo inference
    parser.add_argument('--mc_inference', action='store_true')
    # mc noise_scale
    parser.add_argument('--mc_noise_scale', default=1e-3, type=float)
    # random mc noise inference
    parser.add_argument('--random_mcnoise_inference', action='store_true')
    # mc monte_carlo_samples
    parser.add_argument('--mc_samples', default=10, type=int)

    parser.add_argument('--ambient', action='store_true') # if considering noise model
    parser.add_argument('--refine', action='store_true') # if considering noise model
    # lambda_refinement
    parser.add_argument('--lambda_refinement', default=0.1, type=float)

    # target_snr_ratio -- # 1/2 for FLAIR and 1/3 for T1, T2 original supervised signal
    parser.add_argument('--target_snr_ratio', default=0, type=float) # 1/2 for FLAIR and 1/3 for T1, T2
    # boolean for using random target snr ratio
    parser.add_argument('--random_target_snr_ratio', action='store_true')

    # noise2score param sigma_annealing
    parser.add_argument('--sigma_annealing', default=2304, type=float) # for M4raw
    parser.add_argument('--sigma_max_noise2score', default=0.5, type=float) # for M4raw
    parser.add_argument('--sigma_min_noise2score', default=0.01, type=float) # for M4raw

    parser.add_argument('--modal_num', default=0, type=int, help='For M4Raw, 0 for T1 | 1 for T2 | 2 for FLAIR')

    ## ema setting
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--ema_decay', default=0.999, type=float)


    ######################################################################################################################
    # Dataset parameters
    # parser.add_argument('--traindata_root', type=str, default='/srv/local/shared/data/fastMRI_half/misc/datasets/CBSD400', help='Path to training data')
    # parser.add_argument('--testdata_root', type=str, default='/srv/local/shared/data/fastMRI_half/misc/datasets/CBSD68', help='Path to test data')

    parser.add_argument('--traindata_root', type=str, default='/srv/local/shared/data/fastMRI_half/misc/datasets/Challenge_Train', help='Path to training data')
    parser.add_argument('--traindata_roots', type=str, nargs='+', default=None,
                        help='Multiple paths to training data directories')
    parser.add_argument('--testdata_root', type=str, default='/srv/local/shared/data/fastMRI_half/misc/datasets/Challenge_Val', help='Path to test data')
    parser.add_argument('--eval_noised_data_path', type=str, default=None, help='Path to pre-computed noisy test images')
    # parser.add_argument('--modal', type=str, default='T1', choices=['T1', 'T2', 'FLAIR'], help='MRI modality')
    
    # Model parameters
    parser.add_argument('--noise_sig', type=float, default=50, help='Noise sigma level')
    parser.add_argument('--patch_num', type=int, default=1, help='Number of patches per image')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    # Sigma annealing parameters
    parser.add_argument('--sig_max', type=float, default=50, help='Maximum sigma for annealing')
    parser.add_argument('--sig_min', type=float, default=0.01, help='Minimum sigma for annealing')

    # Other parameters
    parser.add_argument('--target_size', type=int, default=256, help='Target Patch Size')
    
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle data')
    parser.add_argument('--eval_freq', type=int, default=5, help='Evaluation frequency')
    parser.add_argument('--augment', type=bool, default=True, help='Data Augmentation')
    # parser.add_argument('--image_output_dir', type=str, default='img_output', help='Directory for output images')
    parser.add_argument('--use_wandb', type=bool, default=True, help='Use wandb for logging')
    parser.add_argument('--train_id', type=str, default='noise2score_run', help='Training run ID')
    parser.add_argument('--rgb', type=bool, default=False, help='Use RGB images (instead of BGR)')
    # use_grayscale flag
    parser.add_argument('--use_grayscale', type=bool, default=False, help='Use RGB images')

    
    parser.add_argument('--DSM', type=bool, default=False, help='Use DSM for Noise2Score')
    parser.add_argument('--use_dsm_weighting', type=bool, default=True, help='Use DSM weighting for Noise2Score')

    parser.add_argument('--enable_full_image', action='store_true', 
                        help='Enable full image denoising with patch-based approach for large images')
    parser.add_argument('--patch_size', type=int, default=384, 
                        help='Size of patches for patch-based processing')
    parser.add_argument('--stride', type=int, default=384, 
                        help='Stride between patches for patch-based processing')
    
    parser.add_argument('--full', action='store_true', 
                        help='Enable full image denoising')
    
    # use_subset flag
    parser.add_argument('--use_subset', type=bool, default=True, help='Use subset of data for training')
    parser.add_argument('--train_subset_fraction', type=float, default=1/12, help='Size of subset for training')
    # test_subset_fraction
    parser.add_argument('--test_subset_fraction', type=float, default=1/1, help='Size of subset for testing')
    parser.add_argument('--val_scene_fraction', type=float, default=0.1,
                        help='Fraction of training scenes reserved for validation in MEF datasets')
    parser.add_argument('--random_exposure_subset', action='store_true',
                        help='Sample a different 5-of-7 exposure subset per training sample for MEF')
    parser.add_argument('--keep_extreme_exposures', action='store_true',
                        help='When sampling MEF frames, always keep the darkest and brightest exposures')
    
    ## setup training environment
    args = parser.parse_args()
    set_random_seed(args.random_seed)

    ## setup training device
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        args.dist = False
        args.rank = -1
        print('Disabled distributed training.')
    else:
        args.dist = True
        init_dist()
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

    args.save_folder = os.path.join(args.save_folder, args.name)
    args.vis_save_dir = os.path.join(args.save_folder,  'vis')
    args.snapshot_save_dir = os.path.join(args.save_folder,  'snapshot')
    log_file_path = args.save_folder + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log'

    if args.rank <= 0:
        if os.path.exists(args.vis_save_dir) == False:
            os.makedirs(args.vis_save_dir)
        if os.path.exists(args.snapshot_save_dir) == False:
            os.mkdir(args.snapshot_save_dir)
        setup_logger(log_file_path)
        args_txt_path = os.path.join(args.save_folder, 'latest_args.txt')
        with open(args_txt_path, 'w') as f:
            for key in sorted(vars(args)):
                f.write(f'{key}: {getattr(args, key)}\n')

    print_args(args)

    cudnn.benchmark = True

    ## train model
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()
