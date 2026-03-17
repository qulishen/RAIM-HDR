# Training and Reproduction Guide — (NTIRE 2026 RAIM Track2)

**Team:** whu-vip  
**Contact:** whuyuanxy@gmail.com / yuanxiyuan@whu.edu.cn  

> **🔔 IMPORTANT:**
> If you only need to perform **model inference and testing** to reproduce our submission results, please refer to the dedicated inference guide: [`README_inference.md`](./README_inference.md).  
> This document is intended for developers or reviewers who wish to verify the complete data preprocessing and model training pipeline from scratch.

---

## 1. Environment Setup

We provide a one-click setup script that creates a Conda environment named `whu_vip` and installs PyTorch (CUDA 11.8) along with all core dependencies, ensuring compatibility with various GPUs (e.g., V100).

Run the following from the `WHU-VIP_FinalSubmission` root directory:
```bash
bash Scripts/setup_env.sh
```

---

## 2. Training Data Preparation

Before training, the raw LDR images and HDR ground truth must be cropped and converted to `.npy` format to maximize I/O throughput.
**Note:** The preprocessing script strictly requires `1.jpg`, `3.jpg`, `5.jpg`, and `HDR.jpg`. If `2.jpg` or `4.jpg` exist in the folder, they will be safely ignored.

**Expected Directory Structure:**
```text
Datasets/train/
├── 001/
│   ├── 1.jpg      # Under-exposed (Required)
│   ├── 3.jpg      # Normal-exposed (Required)
│   ├── 5.jpg      # Over-exposed (Required)
│   ├── 2.jpg      # (Optional/Ignored)
│   ├── 4.jpg      # (Optional/Ignored)
│   └── HDR.jpg    # Ground Truth (Required)
├── 002/
│   └── ...
```

**Run the data preparation script:**
```bash
conda activate whu_vip

python prepare_dataset.py --source_dir "./Datasets/train" --target_dir "./Train_accelerate/datas" --patch_size 512 --stride 256 --num_workers 8
```

---

## 3. Model Training

Our framework employs a rigorous **4-Stage Curriculum Learning** strategy:
- **Stage 1 (Baseline):** Train from scratch to establish the foundation for feature alignment and fusion.
- **Stage 2 (Strong Aug):** Introduce strong data augmentations (local exposure perturbation, slight spatial misalignment, channel shuffle, and gamma perturbation) to improve generalization in dynamic scenes.
- **Stage 3 (LPIPS):** Enable enhanced perceptual loss weighting to optimize high-frequency textures and overall visual quality.
- **Stage 4 (STE Quant):** Enable STE (Straight-Through Estimator) 8-bit quantization-aware training to effectively mitigate numerical truncation errors when saving the final images.

**One-click execution of the full training pipeline:**
```bash
bash Scripts/train_all_stages.sh
```

### Hardware Allocation & VRAM Usage
1. **Default Parallel Setup:** The script defaults to using **4 NVIDIA RTX 3090 GPUs** with a `batch_size=3` per GPU, resulting in a global effective Batch Size of 12.
2. **VRAM Consumption:** Under the **256×256** random crop input size, each image patch consumes ~7GB of VRAM. The default configuration requires ~21GB of VRAM per GPU.
3. **Low-VRAM Adaptation:** If VRAM is limited, please decrease the `batch_size` and proportionally increase the `grad_accum_steps` (gradient accumulation steps) within the configuration files (`Train_accelerate/configs/*.yaml`). The underlying framework will automatically handle the accumulation to maintain the effective Batch Size.
