# HDR Reconstruction Project (NTIRE 2026 The 3rd Restore Any Image Model (RAIM): Multi-Exposure Image Fusion in Dynamic Scenes (Track 2))

本项目针对NTIRE 2026 The 3rd Restore Any Image Model (RAIM): Multi-Exposure Image Fusion in Dynamic Scenes (Track 2)。

🔗 比赛链接: [https://www.codabench.org/competitions/12728/](https://www.codabench.org/competitions/12728/)

本项目提供了一套高效的 HDR 重建方案。针对服务器环境进行了深度优化，包含一键式环境配置、离线多尺度数据预处理、以及基于 DDP 的渐进式训练流水线。

---

## 📂 项目结构

请确保解压后的文件目录如下，以保证脚本路径引用正确：

```text
. (Project Root)
├── README.md               # 英文版说明文件
├── README_cn.md            # 中文版说明文件
├── env_setup.sh            # 环境一键安装脚本
├── eval.py                 # 快速推理脚本
├── requirements.txt        # Python 依赖清单
├── train.py                # DDP 训练核心脚本
├── train_and_eval.sh       # 完整训练流水线
├── checkpoint/             # 后训练模型存放处
├── DataLoader/             # 数据加载与预处理
├── dataset/                # 数据集根目录
│   ├── testdata_phase2/    # 第二阶段测试集
│   ├── testdata_phase3/    # 第三阶段测试集
│   └── train/              # 训练集
│       ├── train/          # 训练集原图
│       └── ...             # 离线分块训练集
├── model_zoo/              # 预训练模型存放处
│   ├── 06.spynet.pth       # 用于对齐模块的预训练权重
│   └── model_best.pth.tar  # 用于 Quick Inference 的关键权重
├── models/                 # 模型架构定义
├── output/                 # 训练阶段部分图像结果预览
├── output_eval/            # 排行榜测试结果
│   ├── output_eval_phase2/ # 第二阶段测试结果
│   └── output_eval_phase3/ # 第三阶段测试结果
└── utils/                  # 其余项目依赖文件
```

---

## 🛠️ 环境配置

重要提示： 本项目所有实验均在 Python 3.9.25 下开发并验证。为了确保版本兼容性，请严格执行以下步骤。

#### 创建并激活环境 (指定 Python 3.9.25)：

```Bash
conda create -n test_environment python=3.9.25 -y
conda activate test_environment
```
#### 进入项目根目录执行自动化环境配置脚本：

```Bash
chmod +x env_setup.sh
sh env_setup.sh
```

---

## 📦 数据集准备
由于数据集体积限制，提交包未包含图像。请在运行脚本前手动构建以下结构：

```Bash
mkdir -p ./dataset/train/train ./dataset/testdata_phase2 ./dataset/testdata_phase3
```
放置数据：将对应的图像文件解压或链接（ln -s）到上述目录中。确保 ./dataset/train/train 下包含用于裁剪的原始训练图片。

---

## 🚀 操作手册

#### 🧪 快速推理与验证
使用预训练权重快速验证环境。本脚本包含智能显存探针，会自动测试 1024 -> 768 -> 512 -> 256 -> 128 -> 64 的最优分块尺寸以防 OOM：
```Bash
python ./eval.py --ckpt_dir ./model_zoo --eval_dir ./output_eval_phase1 --test_root /home/notebook/data/personal/S9059954/HDR-Competition/datasets/test-stage1-wo-gt --name model_best.pth.tar
```
模型预测结果将保存在 ./output_eval_phase3 中。为了保证能够准确复现排行榜分数，请确保至少能够实现768分块(需要30GiB的显存)。获取的结果需要另外使用在线评分工具进行评分。

#### 🏋️ 完整训练流程
执行集成 Shell 脚本。该脚本会自动串联执行：多尺度离线裁剪 -> 4-GPU DDP 渐进式训练 -> 最终评估。
```sh
sh train_and_eval.sh
```
为了获取能够准确复现排行榜分数的后训练权重，请确保使用至少4张具有至少40GiB显存的显卡。

#### 💡 技术细节与常见问题

- 自适应显存探测：eval.py 内部实现了梯度降级探测机制，即使在 8GB VRAM (如 RTX 5060) 环境下也能自动选择最大可行 Tile Size 运行。
- 分布式配置：训练脚本默认使用 accelerate launch --multi_gpu --num_processes 4。若您的 GPU 数量不为 4，请手动修改 train_and_eval.sh 中的进程数；若您的 GPU 数量为1，请手动删除--multi_gpu。
- 显存优化：针对 768 尺度训练，脚本默认开启了 mixed_precision="bf16" 和 gradient_checkpointing。
- OpenCV 报错 (libGL.so.1)：已通过 opencv-python-headless 解决。请勿在当前环境中安装标准版 opencv-python。
- BasicSR 依赖：由于 tb-nightly 可能导致环境配置错误，env_setup.sh 使用了 --no-deps 策略进行安装，这不会影响模型推理逻辑。

---

祝复现顺利！