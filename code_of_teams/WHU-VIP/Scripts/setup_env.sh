#!/bin/bash
# ==============================================================================
# NTIRE 2026 RAIM (Track 2) - Environment Setup Script
# Team Name: whu_vip
# ==============================================================================

# 自动跳转到项目根目录以便找到 requirements.txt
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=================================================="
echo "Step 1: Setting up Conda Environment [whu_vip]..."
echo "=================================================="
eval "$(conda shell.bash hook)"

if ! { conda env list | grep 'whu_vip'; } >/dev/null 2>&1; then
    conda create -n whu_vip python=3.10 -y
fi

conda activate whu_vip

echo "=================================================="
echo "Step 2: Installing PyTorch (Optimized for V100 GPU)"
echo "=================================================="
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

echo "=================================================="
echo "Step 3: Installing Core Dependencies..."
echo "=================================================="
pip install -r requirements.txt

echo "=================================================="
echo "Environment setup completed successfully!"
echo "You can now run 'bash Scripts/test.sh' or 'bash Scripts/train_all_stages.sh'"
echo "=================================================="