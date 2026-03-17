import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# 禁用 NCCL P2P 和 IB，防止在某些集群环境下多卡训练卡死
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# 导入配置和数据加载器
from option_3 import get_args
from datasets.loader import HDRDataset
from model_interface3 import DehazeModelInterface


class HDRDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset_dir = os.path.join(args.data_dir, args.dataset)

    def setup(self, stage=None):
        # 严格按照 HDRDataset 的参数签名传参，使用关键字传参防止出错
        self.train_dataset = HDRDataset(
            data_dir=self.dataset_dir,
            mode='train',
            size=self.args.patch_size,
            edge_decay=self.args.edge_decay,
            only_h_flip=self.args.only_h_flip
        )

        # 验证集/测试集
        self.val_dataset = HDRDataset(
            data_dir=self.dataset_dir,
            mode='test',  # 假设你的测试文件夹叫 test
            size=self.args.patch_size
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.args.num_workers,
            pin_memory=True
        )


def main():
    # 1. 获取参数
    args = get_args()
    pl.seed_everything(args.seed)

    # 2. 初始化数据模块
    data_module = HDRDataModule(args)

    # 3. 初始化模型 Interface
    # 注意：请确保你的 DehazeModelInterface 内部已经修改为 15 通道输入，
    # 并且在 validation_step 中计算并 log 了 'valid_score'
    model = DehazeModelInterface(args)

    # 4. 回调函数 (Checkpoint & LR Monitor)
    # [关键修改]：以比赛的综合得分 (valid_score) 为准，保存最高分模型
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, args.exp),
        # 文件名示例: model-step050000-score85.50.ckpt
        filename=f'{args.model}-{{step:06d}}-{{valid_score:.2f}}',
        monitor='valid_score',  # 监控比赛综合得分
        mode='max',  # 分数越高越好
        save_top_k=1,  # 只保存分数最高的 1 个模型
        save_last=True  # 始终保存最新的一个 epoch 用于断点续训
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 5. Logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.exp,
        version=args.model
    )

    # 6. Trainer 设置 (基于 Step)
    gpus = [int(x) for x in args.gpu.split(',')]

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=gpus,
        strategy='ddp_find_unused_parameters_true',

        # 关键设置：基于 Step 训练
        max_steps=args.max_steps,
        max_epochs=-1,
        # val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=1,
        # check_val_every_n_epoch=None,
        # limit_train_batches=args.batches_per_epoch,

        precision=args.precision,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        benchmark=True
    )

    # 7. 开始训练
    print(f'==> Start training: {args.model} for {args.max_steps} steps')
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_path)

    print('==> Start testing with best model...')
    # 测试时会自动加载 valid_score 最高的那个 ckpt
    trainer.test(model, datamodule=data_module, ckpt_path='best')


if __name__ == '__main__':
    main()
