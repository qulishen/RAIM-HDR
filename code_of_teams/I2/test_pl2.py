import os
import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# [安全修复] 允许 PyTorch 2.6+ 加载包含 argparse.Namespace 的 Checkpoint
torch.serialization.add_safe_globals([argparse.Namespace])

# 禁用 NCCL P2P 和 IB
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import time
# 导入配置和数据加载器
from option_3 import get_args
from datasets.loader import HDRDataset, HDRTestDataset
from model_interface3 import DehazeModelInterface


# 复用 DataModule (为了保持 test.py 独立，这里保留了精简版的 DataModule)
class HDRTestDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset_dir = os.path.join(args.data_dir, args.dataset)
        self.test_dir = os.path.join(args.data_dir, args.dataset, args.phase)

    # def setup(self, stage=None):
    #     # 测试阶段只需要加载测试集
    #     self.val_dataset = HDRDataset(
    #         data_dir=self.dataset_dir,
    #         mode='test',  # 确保你的测试文件夹叫 test
    #         size=self.args.patch_size
    #     )
    #
    def setup(self, stage=None):
        # [修改这里] 使用专门的无 GT 测试集类
        self.val_dataset = HDRTestDataset(
            data_dir=self.test_dir
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,  # 测试时必须是 1，保证图片一张张保存
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=False  # 测试集不需要打乱
        )


def main():
    # 1. 获取参数
    args = get_args()
    pl.seed_everything(args.seed)

    # ==========================================
    # ⚠️ 请在这里指定你要测试的 Checkpoint 路径！
    # ==========================================
    # 你也可以把这个加到你的 option.py 里通过命令行传入
    # ckpt_path = getattr(args, 'ckpt_path', None)
    # if ckpt_path is None:
    #     # 如果命令行没传，请手动修改这里的路径
    #     ckpt_path = "./experiments/your_model_name/best_model.ckpt"
#    ckpt_path = "/data01/wangjuan/NTIRE/HDR/DehazeFormer-main/saved_models/HDR3/dehazeformer-s-step=113913-valid_score=47.72.ckpt"
    ckpt_path = args.test_ckpt_path
    assert os.path.exists(ckpt_path), f"找不到权重文件: {ckpt_path}，请检查路径！"
    print(f"==> 正在加载模型权重: {ckpt_path}")

    # 2. 初始化数据模块
    data_module = HDRTestDataModule(args)

    # 3. 解析可用设备，并用于 checkpoint 映射与 Trainer 配置
    requested_gpus = [int(x) for x in args.gpu.split(',') if x.strip()]
    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        valid_gpus = [gid for gid in requested_gpus if 0 <= gid < cuda_count]
        if not valid_gpus:
            print(f"[WARN] 请求的 GPU {requested_gpus} 不可用，自动回退到 GPU 0。")
            valid_gpus = [0]
        map_location = torch.device(f"cuda:{valid_gpus[0]}")
        accelerator = 'gpu'
        devices = valid_gpus
    else:
        print("[WARN] 未检测到 CUDA，自动使用 CPU 推理。")
        map_location = torch.device("cpu")
        accelerator = 'cpu'
        devices = 1

    # 3. 从 Checkpoint 加载模型
    # load_from_checkpoint 会自动读取 ckpt 里的权重并实例化你的网络
    model = DehazeModelInterface.load_from_checkpoint(
        ckpt_path, args=args, map_location=map_location, strict=False
    )
    model.eval()  # 切换到测试模式 (关闭 Dropout, BatchNorm 等)

    # 4. 初始化极简版 Trainer
    # 测试阶段不需要 logger 和 callbacks
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        logger=False,  # 关闭 TensorBoard 记录
        benchmark=True
    )

    # 5. 开始测试！
    # 这会自动调用 data_module.test_dataloader() 和 model.test_step()
    print('==> 开始进行 HDR 图像融合与保存...')
    start_time = time.time()
    trainer.test(model, datamodule=data_module)
    end_time = time.time()
    used_time = end_time - start_time
    print("运行时间是",used_time/100)

    # 图片保存的路径由 model_interface.py 中的 test_step 决定
    save_dir = os.path.join('results', args.exp, args.model)
    print(f'==> 测试完成！融合后的 HDR 图像已保存至: {save_dir}')


if __name__ == '__main__':
    main()
