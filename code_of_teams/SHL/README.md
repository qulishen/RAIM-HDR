# 目录说明

```shell

├── basicsr
├── checkpoints # 保存的权重
├── datasets # 训练数据，需要训练请将数据放在此处（参考下文）
├── options # 配置文件  
├── postdeal.py # 后处理/模型集成代码
├── README.md
├── readme.txt # 提交时zip打包进的说明文件
├── requirements.txt
├── test.sh 
├── ...

```

# 训练和推理环境  
- Ubuntu 22.04.4
- 1 个NVIDIA GeForce RTX 4090(24G)
- cuda-12.8
- torch 2.8.0+cu128
- Python 3.10.18   

其他库参考requirements.txt.

# 推理

进入到代码目录（`SHL_RAIM-HDR/`），修改`test.sh`中3处测试数据路径:`--input_dir`
（如：阶段3 datasets/testdata_phase3/，阶段2 datasets/testdata-phase2/，注意下划线中划线）：
```shell
bash test.sh
```

文件夹下`output_mean`为最终输出.

推理时间，其他信息参见`readme.txt`：

```shell
uformer：238秒
restormer：127秒
mstpp：332秒
postdeal: 20秒
合计（顺序执行）： 7.17秒/张
```

# 训练

## 数据
1. 仅使用官方提供的训练数据，未使用额外数据集。数据存放如下：
```shell
datasets/
├── testdata-phase2
│   ├── 001
│   ├── 002
│   ├── ...
├── testdata-phase2.zip
├── testdata_phase3
│   ├── 001
│   ├── 002
│   ├── ...
├── testdata_phase3.zip
├── train
│   ├── 001
│   ├── 002
│   ├── ...
└── train.zip

```

2. 模型都未使用预训练权重.

## 运行训练
```shell
python setup.py develop
```

第一次迭代：
```shell
CUDA_VISIBLE_DEVICES=1 python basicsr/train.py -opt options/Uformer.yml --auto_resume
CUDA_VISIBLE_DEVICES=1 python basicsr/train.py -opt options/restormer.yml --auto_resume
CUDA_VISIBLE_DEVICES=1 python basicsr/train.py -opt options/mstpp.yml --auto_resume
```

修改`option/*_step2.yml`中的`network_g->pretrained`权重路径为上一步迭代的最佳权重路径。执行第二次迭代：
```shell
CUDA_VISIBLE_DEVICES=1 python basicsr/train.py -opt options/mstpp_step2.yml --auto_resume
CUDA_VISIBLE_DEVICES=1 python basicsr/train.py -opt options/restormer_step2.yml --auto_resume
CUDA_VISIBLE_DEVICES=1 python basicsr/train.py -opt options/Uformer_step2.yml --auto_resume
```

权重和日志保存在`experiments`和`tb_logger`中。

# 其他
队伍：SHL(snowli  lishihang94@qq.com)     
联系方式（微信同号）：13267052013