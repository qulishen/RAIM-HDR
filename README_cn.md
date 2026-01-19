<p align="center">
  <img src="logo.png" alt="RAIM-HDR Logo" width="200"/>
</p>

<h1 align="center">🌟 RAIM-HDR: 动态场景下的多曝光图像融合 </h1>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/Language-English-blue" alt="English"></a>
  <a href="README_cn.md"><img src="https://img.shields.io/badge/语言-中文-red" alt="中文"></a>
  <a href="https://www.codabench.org/competitions/12728/"><img src="https://img.shields.io/badge/🏆_比赛-CodaBench-orange" alt="Competition"></a>
</p>

---

本仓库提供了 **第三届 Restore Any Image Model (RAIM) 挑战赛：动态场景下的多曝光图像融合 (赛道2)** 的基线代码 🎯

🔗 **比赛链接**: [https://www.codabench.org/competitions/12728/](https://www.codabench.org/competitions/12728/)

## 📖 概述

本赛道的目标是将多张不同曝光级别的 LDR（低动态范围）图像融合成一张 HDR（高动态范围）图像，同时处理动态场景中可能出现的运动伪影。✨

我们的基线模型基于 **Uformer** 架构，针对多曝光图像融合任务进行了适配。🚀

## 📦 数据集

### 📥 下载

从以下链接下载数据集：[www.example.com](https://www.example.com)

### 🗂️ 目录结构

数据集应按以下方式组织：

```
datasets/
├── train/
│   ├── 001/
│   │   ├── 0.jpg      # 🌑 最暗曝光
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── 3.jpg      # ⚖️ 中等曝光
│   │   ├── 4.jpg
│   │   ├── 5.jpg
│   │   └── 6.jpg      # ☀️ 最亮曝光
│   │   └── HDR.jpg    # 🎨 HDR 图像
│   ├── 002/
│   │   ├── 0.jpg
│   │   ├── ...
│   │   └── 6.jpg
│   │   └── HDR.jpg    # 🎨 HDR 图像
│   └── ...
└── test/
    ├── 001/
    │   ├── 1.jpg
    │   ├── ...
    │   └── 5.jpg
    └── ...
```

训练集的每个场景文件夹包含 7 张图像（0.jpg 到 6.jpg），按曝光级别从暗到亮排序。📸

## 🛠️ 安装

### 📋 环境要求

- 🐍 Python >= 3.8
- 🔥 PyTorch >= 1.12
- ⚡ CUDA >= 11.3

### 📦 安装依赖

```bash
pip install -r requirements.txt
```

训练所需的额外依赖：

```bash
pip install opencv-python-headless timm scipy einops accelerate lmdb ftfy tqdm Pillow tensorboard
```

## 🚀 使用方法

### 🏋️ 训练

使用提供的脚本开始训练：

```bash
./train.sh
```

或者使用自定义设置手动运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=4392 \
    basicsr/train.py -opt options/Uformer.yml --launcher pytorch --auto_resume
```

**关键训练参数**（在 `options/Uformer.yml` 中配置）：

| 参数                 | 值     | 描述           |
| -------------------- | ------ | -------------- |
| `batch_size_per_gpu` | 8      | 每个 GPU 的批大小 |
| `gt_size`            | 256    | 训练图块大小   |
| `total_iter`         | 300000 | 总训练迭代次数 |
| `lr`                 | 2e-4   | 学习率         |
| `ema_decay`          | 0.999  | EMA 衰减率     |

### 🧪 测试 / 推理

在测试图像上运行推理：

```bash
python test_crop.py \
    --input_dir /path/to/test/images \
    --output_dir /path/to/output \
    --weights /path/to/checkpoint.pth \
    --opt options/Uformer.yml \
    --crop_size 256 \
    --crop_shave 32 \
    --factor 16
```

**参数说明：**

| 参数           | 默认值 | 描述                     |
| -------------- | ------ | ------------------------ |
| `--input_dir`  | -      | 包含序列子文件夹的根目录 |
| `--output_dir` | -      | 融合后 HDR 图像的输出目录 |
| `--weights`    | -      | 训练好的模型检查点路径   |
| `--opt`        | -      | YAML 配置文件路径        |
| `--crop_size`  | 256    | 分块推理的块大小         |
| `--crop_shave` | 32     | 分块混合的重叠大小       |
| `--factor`     | 16     | 网络步长的填充因子       |

### 📊 评估

对比真实值评估预测结果：

```bash
python eval.py \
    --pred_root /path/to/predictions \
    --gt_root /path/to/ground_truth \
    --device auto
```

**参数说明：**

| 参数          | 默认值 | 描述               |
| ------------- | ------ | ------------------ |
| `--pred_root` | -      | 包含预测图像的目录 |
| `--gt_root`   | -      | 包含真实图像的目录 |
| `--device`    | auto   | 设备选择 (auto/cpu/cuda) |

**评估指标：** 📈

- 📐 **PSNR**: 峰值信噪比
- 🔍 **SSIM**: 结构相似性指数
- 🧠 **LPIPS**: 学习感知图像块相似度
- 🏆 **Score**: 遵循 NTIRE 公式的加权组合得分

## 🎬 示例与挑战重点

<p align="center">
  <img src="example.png" alt="HDR 融合示例" width="100%"/>
</p>

上图展示了多曝光 HDR 融合中的关键挑战。参赛选手可以从 **两个主要方面** 来提升模型性能：

### 🔴 动态范围恢复（红框区域）

| 挑战 | 描述 |
|------|------|
| 🌑 **欠曝光** | 暗区细节丢失，噪声明显 |
| ☀️ **过曝光** | 亮区饱和，细节被冲掉 |
| 🎯 **目标** | 恢复完整的动态范围，使阴影和高光区域都具有丰富的细节 |

### 🟡 运动鬼影消除（黄框区域）

| 挑战 | 描述 |
|------|------|
| 👻 **鬼影伪影** | 运动物体出现半透明的重影 |
| 🏃 **运动模糊** | 快速移动的主体边缘模糊 |
| 🎯 **目标** | 生成无鬼影的 HDR 图像，保持运动物体的清晰边缘 |

## 📁 项目结构

```
RAIM-HDR/
├── basicsr/                 # 🔧 核心库
│   ├── archs/              # 🏗️ 网络架构 (Uformer 等)
│   ├── data/               # 📊 数据集加载器
│   ├── losses/             # 📉 损失函数
│   ├── metrics/            # 📏 评估指标
│   ├── models/             # 🤖 训练模型
│   ├── ops/                # ⚙️ 自定义 CUDA 算子
│   ├── utils/              # 🛠️ 工具函数
│   ├── train.py            # 🏋️ 训练脚本
│   └── test.py             # 🧪 测试脚本
├── datasets/               # 📦 数据集文件夹
├── options/                # ⚙️ 配置文件
│   └── Uformer.yml        # 📝 Uformer 训练配置
├── train.sh               # 🚀 训练启动脚本
├── test_crop.py           # 🔬 分块推理脚本
├── eval.py                # 📊 评估脚本
├── requirements.txt       # 📋 Python 依赖
└── README.md
```

## 📄 许可证

本项目仅供学术研究使用。🎓

## 🙏 致谢

- [BasicSR](https://github.com/XPixelGroup/BasicSR) - 训练框架 💪
- [Uformer](https://github.com/ZhendongWang6/Uformer) - 骨干网络架构 🌟

---

<p align="center">
  <b>祝比赛顺利！🍀🎉</b>
</p>
