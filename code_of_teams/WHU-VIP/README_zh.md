# 训练与复现指南 —（NTIRE 2026 RAIM Track2）

**队伍：** whu-vip  
**联系方式：** whuyuanxy@gmail.com / yuanxiyuan@whu.edu.cn  

> **🔔 重要提示：**
> 如果您仅需要进行**模型推理与测试 (Inference)**，请直接查阅推理专用文档：[`README_inference.md`](./README_inference.md) 或 [`README_inference_zh.md`](./README_inference_zh.md)。  
> 本文档面向需要复现完整数据预处理与模型训练流程的开发者。

---

## 1. 环境配置

我们提供了一键配置脚本，自动创建 `whu_vip` Conda 环境，并安装兼容各类显卡（如 V100）的 PyTorch (CUDA 11.8) 及核心依赖。

请在 `WHU-VIP_FinalSubmission` 根目录下运行：
```bash
bash Scripts/setup_env.sh
```

---

## 2. 制作训练集数据

训练前需将原始 LDR 图像和 HDR 标签裁剪为 `.npy` 格式以提升 I/O 吞吐率。
**注意：** 数据预处理仅强依赖 `1.jpg`、`3.jpg`、`5.jpg` 和 `HDR.jpg`，若场景中存在 `2.jpg` 或 `4.jpg`，脚本会自动忽略。

**训练数据目录结构示例：**
```text
Datasets/train/
├── 001/
│   ├── 1.jpg      # 欠曝光 (必选)
│   ├── 3.jpg      # 正常曝光 (必选)
│   ├── 5.jpg      # 过曝光 (必选)
│   ├── 2.jpg      # (可选/忽略)
│   ├── 4.jpg      # (可选/忽略)
│   └── HDR.jpg    # 真值 (必选)
├── 002/
│   └── ...
```

**运行数据准备脚本：**
```bash
conda activate whu_vip

python prepare_dataset.py --source_dir "./Datasets/train" --target_dir "./Train_accelerate/datas" --patch_size 512 --stride 256 --num_workers 8
```

---

## 3. 模型训练

本方案采用 **4 阶段课程学习 (Curriculum Learning)** 策略：
- **Stage 1 (Baseline):** 从零开始收敛，建立特征对齐与融合基础。
- **Stage 2 (Strong Aug):** 引入强数据增强（局部曝光扰动、轻微空间错位、通道洗牌、Gamma扰动），提升模型在动态场景下的泛化能力。
- **Stage 3 (LPIPS):** 开启感知损失权重增强，优化图像的高频纹理与视觉观感。
- **Stage 4 (STE Quant):** 开启 STE (Straight-Through Estimator) 8-bit 量化感知训练，有效缓解最终保存图像时的数值截断误差。

**一键运行完整训练流：**
```bash
bash Scripts/train_all_stages.sh
```

### 硬件分配与显存说明
1. **默认并行配置：** 脚本默认使用 **4张 NVIDIA RTX 3090 显卡**，每张卡 `batch_size=3`，全局等效 Batch Size 为 12。
2. **显存消耗评估：** 在 **256×256** 的随机裁剪输入尺寸下，每张图片（Patch）约占用 7GB 显存。默认配置下，单卡约需 21GB 显存。
3. **低显存硬件适配策略：** 若显卡显存不足，请在配置 (`Train_accelerate/configs/*.yaml`) 中降低 `batch_size`，并等比例调大 `grad_accum_steps` (梯度累加步数)，底层框架将自动接管以保持等效 Batch Size 不变。
