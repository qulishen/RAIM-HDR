# NTIRE 2026 RAIM MEF - Track 2: Multi-Exposure Image Fusion

## Team Information
- **Team:** miketjc
- **Members:** Guoyi Xu, Yaoxin Jiang, Cici Liu, Yaokun Shi, Jiachen Tu
- **Contact:** jtu9@illinois.edu
- **Institution:** University of Illinois Urbana-Champaign

---

## Performance

| Phase | Score | Method |
|-------|-------|--------|
| Phase 2 (Validation) | **56.6913 PSNR** | Composite loss (L1 + SSIM + LPIPS, λp=0.8) |
| Phase 3 (Final Test) | **57.1426 PSNR** | stride-128 + 8-way ensemble |

---

## Package Contents

This package includes everything needed for training and inference:

### Training Code
- `train.py` - Main training script
- `models/trainer.py` - Trainer class with loss functions
- `models/losses.py` - Composite loss implementation (L1 + SSIM + LPIPS)
- `models/archs/TimeDiffiT_ResNet_color_128_arch.py` - Model architecture
- `dataloader/dataset_mef.py` - MEF dataset loader
- `train.sh` - Training shell script

### Inference Code
- `scripts/raim_mef/infer_mef.py` - Core inference implementation
- `infer.sh` - Easy inference script

### Model Checkpoint
- `checkpoints/raim_mef_phase2j/snapshot/net_final.pth` (544 MB)
  - Phase 2 validation score: 56.6913 PSNR
  - Phase 3 test score: 57.1426 PSNR

### Documentation
- `README.md` - This file
- `SUBMISSION_GUIDE.md` - Submission checklist and email template
- `requirements.txt` - All Python dependencies

### Supporting Materials
- `logs/raim_mef_phase2j_train.log` - Training log
- `logs/PHASE3_EXPERIMENTS.md` - Phase 3 experiment details
- `submissions/` - Phase 3 proof of results (3 submission ZIPs)

---

## System Requirements

- **GPU:** NVIDIA GPU with CUDA 12.1 support 
- **CPU:** 16+ cores recommended
- **RAM:** 64GB+
- **CUDA:** 12.1
- **Python:** 3.10+
- **OS:** Linux

---

## Installation

### Step 1: Create Environment

```bash
conda create -n raim_mef python=3.10 -y
conda activate raim_mef
```

### Step 2: Install PyTorch

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

Run the test script to verify everything is set up correctly:

```bash
./test.sh
```

This will check:
- Python version
- All dependencies (PyTorch, OpenCV, LPIPS, einops)
- GPU availability
- Model checkpoint integrity
- Model loading capability

---

## Training

### Quick Start

```bash
# Basic training (Phase 2j: 300 epochs, λp=0.8)
./train.sh
```

### Custom Training

```bash
python train.py \
    --data_path /path/to/TrainSet_MEF \
    --save_folder /path/to/save \
    --batch_size 4 \
    --patch_size 256 \
    --epochs 300 \
    --lr 1e-5 \
    --loss_composite True \
    --lambda_lpips 0.8
```

### Resume from Checkpoint

```bash
python train.py \
    --resume /path/to/checkpoint.pth \
    --epochs 300 \
    --lr 1e-5
```

### SLURM (HPC Clusters)

```bash
# ICC cluster
sbatch scripts/raim_mef/slurm_train_composite.sh

# Lab servers
bash scripts/raim_mef/train_phase2d_lab.sh
```

---

## Inference

### Quick Start

```bash
# Run inference on 100 test scenes
./infer.sh
```

### Custom Inference

```bash
python scripts/raim_mef/infer_mef.py \
    --model_path checkpoints/raim_mef_phase2j/snapshot/net_final.pth \
    --input_dir /path/to/testdata-phase3/testdata_phase3 \
    --output_dir ./results \
    --stride 128 \
    --ensemble \
    --tile_size 512 \
    --jpeg_quality 95
```

### Parameters

- `--stride` - Tiling stride (64, 96, 128, 256)
  - Smaller = higher quality, slower
  - Default: 256 (fast)
  - Recommended: 128 (best quality/speed balance)

- `--ensemble` - 8-way geometric ensemble (rotation + flip)
  - Recommended for best results
  - Adds ~8× computation time

- `--tile_size` - Maximum tile size (256, 512)
  - Larger tiles = faster but require more VRAM

- `--jpeg_quality` - JPEG compression quality (0-100)
  - Recommended: 95

### Expected Performance

- **Inference Time** (100 scenes):
  - stride-256: ~15 min
  - stride-128: ~1 hour
  - stride-128+ensemble: ~6.5 hours

- **Output:**
  - 100 JPEG images (2040×1528 pixels)
  - Total size: ~59 MB (stride-256), ~120 MB (stride-128+ens)
  - Filename format: 001.jpg, 002.jpg, ..., 100.jpg

---

## Architecture

**Model:** TimeDiffiT_ResNet_color_128
- **Parameters:** 142.5M
- **Input:** 5 multi-exposure images (2040×1528) → 15-channel tensor
- **Output:** 1 fused HDR image (2040×1528 JPEG)
- **Loss:** Composite (L1 + SSIM + LPIPS)
  - λ_l1 = 1.0
  - λ_ssim = 1.0
  - λ_lpips = 0.8 (Phase 2j)

---

## Datasets

### Training Dataset

The training dataset (TrainSet_MEF) is provided by the NTIRE 2026 challenge organizers:

**Download:** [https://drive.google.com/drive/folders/1LJe22KD7GhILJ9dGAw6eIgTQpSEfJENp?usp=sharing](https://drive.google.com/drive/folders/1LJe22KD7GhILJ9dGAw6eIgTQpSEfJENp?usp=sharing)

- **Scenes:** 90 training scenes
- **Format:** 5 multi-exposure images per scene + ground truth fused image
- **Size:** ~15 GB

### Validation/Test Datasets

- **TestSet_MEF:** 100 validation scenes (from NTIRE track)
- **Phase 3 Test Data:** 100 blind test scenes (from NTIRE phase 3)

### Dataset Format

#### Training Dataset Structure

```
TrainSet_MEF/
├── 001/
│   ├── exp_-2.0.jpg     # -2 stops exposure
│   ├── exp_-1.0.jpg     # -1 stop
│   ├── exp_0.0.jpg      # reference
│   ├── exp_+1.0.jpg     # +1 stop
│   ├── exp_+2.0.jpg     # +2 stops
│   └── gt.jpg           # ground truth (fused image)
├── 002/
│   └── ...
└── ... (90 scenes total)
```

### Inference Dataset Structure

```
testdata_phase3/
├── 001/
│   ├── exp_-2.0.jpg
│   ├── exp_-1.0.jpg
│   ├── exp_0.0.jpg
│   ├── exp_+1.0.jpg
│   └── exp_+2.0.jpg
├── 002/
│   └── ...
└── ... (100 scenes)
```

---

## Troubleshooting

### Out of Memory (OOM)

- Reduce `--tile_size` (256 → 128)
- Reduce `--stride` (larger strides = smaller tiles)
- Use fewer GPUs (set `CUDA_VISIBLE_DEVICES`)

### Slow Inference

- Increase `--stride` (128 → 256)
- Remove `--ensemble` flag
- Use `--tile_size 512` for faster processing

### Model Loading Fails

- Verify checkpoint path exists
- Ensure PyTorch 2.5.1 is installed
- Check CUDA compatibility (requires CUDA 12.1)

### Import Errors

- Verify all dependencies installed: `pip install -r requirements.txt`
- Ensure activated conda environment: `conda activate raim_mef`
- Check Python version: `python --version` (should be 3.10+)

---

## Support

For questions or issues:
- **Email:** jtu9@illinois.edu
- **Response time:** Within 24 hours
- **Available through:** March 16, 2026

---

**Team:** miketjc | **Phase 4 Submission** | **March 12, 2026**
