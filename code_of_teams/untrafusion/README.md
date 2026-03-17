# HDR Reconstruction Project (NTIRE 2026 The 3rd Restore Any Image Model (RAIM): Multi-Exposure Image Fusion in Dynamic Scenes (Track 2))

This project is developed for the NTIRE 2026 The 3rd Restore Any Image Model (RAIM): Multi-Exposure Image Fusion in Dynamic Scenes (Track 2).

🔗 **Challenge Link:** [https://www.codabench.org/competitions/12728/](https://www.codabench.org/competitions/12728/)

This repository provides a high-performance HDR reconstruction solution. It is deeply optimized for server environments, featuring a one-click environment setup, offline multi-scale data preprocessing, and a progressive training pipeline based on Distributed Data Parallel (DDP).

---

## 📂 Project Structure

Please ensure the unzipped file structure matches the following to ensure correct script path references:

```text
. (Project Root)
├── README.md               # English instruction file
├── README_cn.md            # Chinese instruction file
├── env_setup.sh            # Automated environment setup script
├── eval.py                 # Fast inference script
├── requirements.txt        # Python dependency list
├── train.py                # Core DDP training script
├── train_and_eval.sh       # Full training pipeline script
├── checkpoint/             # Directory for post-training models
├── DataLoader/             # Data loading and preprocessing modules
├── dataset/                # Dataset root directory
│   ├── testdata_phase2/    # Test set for Phase 2
│   ├── testdata_phase3/    # Test set for Phase 3
│   └── train/              # Training set root
│       ├── train/          # Original training images
│       └── ...             # Offline patched training sets
├── model_zoo/              # Pre-trained model directory
│   ├── 06.spynet.pth       # Pre-trained weights for the alignment module
│   └── model_best.pth.tar  # Key weights for Quick Inference
├── models/                 # Model architecture definitions
├── output/                 # Image result previews during training
├── output_eval/            # Leaderboard test results
│   ├── output_eval_phase2/ # Phase 2 results
│   └── output_eval_phase3/ # Phase 3 results
└── utils/                  # Other project utility dependencies
```

## 🛠️ Environment Setup

Important Note: All experiments were developed and verified under Python 3.9.25. To ensure version compatibility, please strictly follow the steps below.

#### Create and Activate Environment (Python 3.9.25):

```Bash
conda create -n test_environment python=3.9.25 -y
conda activate test_environment
```

#### Run Automated Setup Script in Project Root:

```Bash
chmod +x env_setup.sh
sh env_setup.sh
```

---

## 📦 Data Preparation

Due to file size constraints, the submission package does not include images. Please manually construct the following structure before running the scripts:

```Bash
mkdir -p ./dataset/train/train ./dataset/testdata_phase2 ./dataset/testdata_phase3
```

Placing Data: Unzip image files or create symbolic links (ln -s) into the directories above. Ensure that ./dataset/train/train contains the original high-resolution training images used for cropping.

---

## 🚀 Execution Guide


#### 🧪 Quick Inference

Use pre-trained weights to verify the environment. This script includes a Smart VRAM Probe that automatically tests the optimal tile size from 1024 -> 768 -> 512 -> 256 -> 128 -> 64 to prevent Out-of-Memory (OOM) errors:

```Bash
python ./eval.py \
    --ckpt_dir ./model_zoo \
    --eval_dir ./output_eval_phase3 \
    --test_root ./dataset/testdata_phase3 \
    --name model_best.pth.tar
```

Predicted results will be saved in ./output_eval_phase3. To ensure accurate reproduction of leaderboard scores, please ensure the system can at least process 768-sized tiles (requires approximately 30GiB VRAM).The results obtained need to be scored using online rating tool.

#### 🏋️ Full Training Pipeline

Execute the integrated shell script. This script sequentially performs: Offline Multi-scale Cropping -> 4-GPU DDP Progressive Training -> Final Evaluation.

```sh
sh train_and_eval.sh
```

To obtain post-training weights that accurately reproduce leaderboard scores, please ensure you use at least 4 GPUs, each with at least 40GiB VRAM.

#### 💡 Technical Details & Troubleshooting

- Adaptive VRAM Probing: eval.py implements a gradient degradation probe. It can automatically select the maximum feasible tile size even in 8GB VRAM environments (e.g., RTX 5060).

- Distributed Configuration: The training script uses accelerate launch --multi_gpu --num_processes 4 by default. If your GPU count is not 4, please manually modify the process count in train_and_eval.sh. If using a single GPU, remove the --multi_gpu flag.

- Memory Optimization: For 768-scale training, mixed_precision="bf16" and gradient_checkpointing are enabled by default.

- OpenCV Error (libGL.so.1): Resolved via opencv-python-headless. Do not install the standard opencv-python in this environment.

- BasicSR Dependencies: To avoid potential environment errors caused by tb-nightly, env_setup.sh uses a --no-deps installation strategy, which does not affect model inference logic.

---

Good luck with the reproduction!


