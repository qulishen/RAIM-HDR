# option.py

import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Image Dehazing Training')

    # ... (前面的基础设置保持不变) ...
    parser.add_argument('--model', default='dehazeformer-s', type=str, help='model name')
    parser.add_argument('--dataset', default='HDR', type=str, help='dataset name')
    parser.add_argument('--exp', default='HDR2', type=str, help='experiment setting name')
    parser.add_argument('--seed', default=2026, type=int, help='random seed')
    parser.add_argument('--num_workers', default=16, type=int, help='number of data loading workers')
    parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')

    # ... (路径设置保持不变) ...
    parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset root')
    parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to save checkpoints')
    parser.add_argument('--log_dir', default='./logs/', type=str, help='path to tensorboard logs')
    parser.add_argument('--resume_path', default=None, type=str, help='加载模型')
    

    # ---------------------------------------------------------
    # 3. 训练超参数 (修改部分)
    # ---------------------------------------------------------
    parser.add_argument('--batch_size', default=8, type=int, help='batch size for training')
    parser.add_argument('--patch_size', default=512, type=int, help='image patch size for training')
    parser.add_argument("--batches_per_epoch",default=30,type=int,help='number of batches in each epoch')

    # [修改] 不再使用 epochs，改为 max_steps
    # 例如：总共训练 300,000 步
    parser.add_argument('--max_steps', default=270000, type=int, help='max number of training steps')

    parser.add_argument('--lr', default=2e-4, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adam', 'adamw'], help='optimizer type')

    # ---------------------------------------------------------
    # 4. 验证与数据增强设置 (修改部分)
    # ---------------------------------------------------------
    # [修改] 验证频率改为按 Step 计算
    # 例如：每 5000 步验证一次
    parser.add_argument('--val_check_interval', default=90, type=int, help='validation frequency in steps')

    parser.add_argument('--valid_mode', default='test', type=str, help='validation dataset mode')
    parser.add_argument('--edge_decay', default=0, type=float, help='edge decay parameter')
    parser.add_argument('--only_h_flip', action='store_true', help='if specified, only use horizontal flip')
    parser.add_argument('--phase', default='testdata_phase2', type=str, help='validation dataset mode')
    parser.add_argument('--test_ckpt_path', default="./saved_models/HDR3/dehazeformer-s-step=113913-valid_score=47.72.ckpt", help='test_ckpt_path')

    # ... (混合精度设置保持不变) ...
    parser.add_argument('--precision', default='16-mixed', type=str, help='training precision (16-mixed or 32)')

    args = parser.parse_args()
    return args
