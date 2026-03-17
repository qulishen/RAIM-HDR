# Inference Guide — (NTIRE 2026 RAIM Track2)

**Team:** whu-vip  
**Contact:** whuyuanxy@gmail.com / yuanxiyuan@whu.edu.cn  
**Architecture:** Based on AFUNet, enhanced with Spatial Feature Transform (SFT) and adaptive Gated Map modules, specifically designed for multi-exposure HDR fusion in dynamic scenes.

---

## 1. Environment Setup (Optimized for V100 GPUs)

To ensure smooth execution in the testing environment (e.g., NVIDIA V100 GPU) without compilation errors, we strongly recommend using PyTorch with **CUDA 11.8**.

We provide a one-click setup script. Please run the following from the `WHU-VIP_FinalSubmission` root directory:
```bash
bash Scripts/setup_env.sh
```

**Alternatively, you can manually execute the following commands:**

```bash
# 1. Create and activate a fresh conda environment
conda create -n whu_vip python=3.10 -y
conda activate whu_vip

# 2. Install V100-compatible PyTorch
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --extra-index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 3. Install core dependencies
pip install -r requirements.txt
```

---

## 2. Pretrained Weights

The optimal model weights required for inference are included in the submission, located at:
```text
PretrainedModels/WHU_VIP_AFUNet_SFT_GM.pth
```

---

## 3. Data Preparation & Input Format

Each test scene should be placed in a subfolder under `--input_root`, named with a 3-digit zero-padded number (e.g., `001` to `100`).  
To perfectly balance computational efficiency and information richness, our model only requires **3 frames** of different exposures per scene to achieve high-quality fusion:

```text
<input_root>/
├── 001/
│   ├── 0.jpg    # Under-exposed frame
│   ├── 2.jpg    # Normal-exposed frame (Reference)
│   └── 4.jpg    # Over-exposed frame
├── 002/
│   └── ...
```

---

## 4. Running Inference

We highly recommend using the provided shell script for one-click inference. Simply open `Scripts/test.sh`, modify the `INPUT_DIR` and `SAVE_DIR` paths at the top, and run:

```bash
bash Scripts/test.sh
```

### Argument Details

| Argument | Default | Description |
|---|---|---|
| `--input_root` | *(Required)* | Root directory of test scenes |
| `--checkpoint` | *(Required)* | Path to the model weights |
| `--save_root` | *(Required)* | Output directory |
| `--gpu_list` | `[0]` | List of GPU IDs to use (supports multi-GPU, e.g., `0 1 2 3`) |
| `--whole_image` | `True` | Whole-image inference flag (auto-fallbacks to split-inference on OOM) |
| `--zip_name` | `submission_whu_vip.zip` | Name of the final submission zip archive |

---

## 5. Inference Strategy & VRAM Usage

This codebase features an extreme VRAM-adaptive safety mechanism:
- **Default (Whole-Image Inference):** The input is padded to a multiple of 128 for a seamless single forward pass. For a 1528×2040 resolution, it consumes **~24 GB VRAM** per GPU (e.g., RTX 3090 / 4090 / A100).
- **Auto-Fallback (Split Inference):** If the program detects a `CUDA Out of Memory` error on GPUs with limited memory (e.g., a 16GB V100), it will **automatically fall back** to a split-inference mode. The image is split into 2 overlapping patches along the long edge (size 1280), inferred separately, and seamlessly blended using a linear gradient mask. Peak VRAM in this mode is strictly kept under **~15 GB**.

---

## 6. Output Details

Upon completion, the following will be generated in the `--save_root` directory:
1. **100 Reconstructed HDR Images**: Named `001.jpg` through `100.jpg`.
2. **readme.txt**: Auto-generated file containing the average runtime per image and architecture description, formatted exactly as requested by the organizers.
3. **Submission Archive (`submission_whu_vip.zip`)**: An automatically packaged zip containing the images and readme.txt, ready for direct Phase 3 submission.

---

## 7. Notes

- **AMP (Automatic Mixed Precision)** in FP16 is enabled by default during inference.
- Multi-GPU parallel inference utilizes the Python `multiprocessing` library with the `spawn` start method. Each GPU is assigned an independent sub-process, ensuring maximum throughput without interference.
- It is strongly recommended to execute all commands from the `WHU-VIP_FinalSubmission/` root directory to ensure correct Python module imports.
