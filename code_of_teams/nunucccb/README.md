# NTIRE 2026 Multi-Exposure Image Fusion in Dynamic Scenes TEAM-nunucccb


## 1. 硬件要求与环境配置 (Hardware & Environment)

我们的实验和模型评估在以下硬件和软件环境中进行：

* **硬件:** NVIDIA L40 GPU (48GB VRAM)
* **操作系统:** Ubuntu 20.04 LTS
* **CUDA 版本:** 12.1 

### 环境安装


```bash
# 1. 创建并激活虚拟环境
conda create -n hdr_env python=3.10 -y
conda activate hdr_env

# 2. 安装 PyTorch 和 Torchvision 
pip install torch torchvision 

# 3. 安装其他依赖
pip install -r requirements.txt
```

## 2. 数据集准备 (Data Preparation)

请将比赛提供的训练集和测试集按照以下目录结构放置在 `datasets/` 文件夹下。

其中，每个子文件夹（如 `001/`）代表一个场景，包含了不同曝光度（LDR）的输入图像以及对应的 Ground Truth (`HDR.jpg`)。

```text
datasets/
├── train/
│   ├── 001/
│   │   ├── 0.jpg      # 🌑 最暗曝光
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── 3.jpg      # ⚖️ 中等曝光
│   │   ├── 4.jpg
│   │   ├── 5.jpg
│   │   ├── 6.jpg      # ☀️ 最亮曝光
│   │   └── HDR.jpg    # 🎨 HDR 图像 (Ground Truth)
│   ├── 002/
│   │   ├── 0.jpg
│   │   ├── ...
│   │   ├── 6.jpg
│   │   └── HDR.jpg    
│   └── ...
└── test/
    ├── 001/
    │   ├── 1.jpg
    │   ├── ...
    │   └── 5.jpg
    └── ...
```

## 3. 模型训练 (Training)


```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/HDR_transformer.yml --auto_resume
```


## 4. 模型推理
请运行以下推理代码。结果将保存在 `./result` 文件夹中：
```bash
bash test.sh
```

## Acknowledgments

* 本项目代码基于官方代码库[RAIM-HDR](https://github.com/qulishen/RAIM-HDR)构建。