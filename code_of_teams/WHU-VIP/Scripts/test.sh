#!/bin/bash
# ==============================================================================
# NTIRE 2026 RAIM (Track 2) - Phase 3 Inference Script
# Team Name: whu_vip
# ==============================================================================

# 自动跳转到项目根目录以便执行 inference.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ==============================================================================
# USER CONFIGURATION (Please modify the paths below as needed)
# ==============================================================================
# The root directory containing the test dataset (e.g., testdata_phase3/)
INPUT_DIR="/home/notebook/data/personal/S9059954/HDR-Competition/datasets/test-stage2"

# Path to our slimmed pretrained model checkpoint
CHECKPOINT_PATH="./PretrainedModels/WHU_VIP_AFUNet_SFT_GM.pth"

# Base directory where the final output images and the zip file will be saved
BASE_SAVE_DIR="./results"

# Specify the GPU IDs to use (space separated). e.g., "0" or "0 1 2 3"
GPU_LIST="0"

# Name of the final zip file
ZIP_NAME="submission_whu_vip.zip"
# ==============================================================================

# 自动提取输入文件夹的名称，并拼接成最终的保存子路径
INPUT_NAME=$(basename "$INPUT_DIR")
FINAL_SAVE_DIR="${BASE_SAVE_DIR}/${INPUT_NAME}_results_WHU_VIP"

echo "=================================================="
echo "Activating Conda Environment [whu_vip]..."
echo "=================================================="
eval "$(conda shell.bash hook)"
conda activate whu_vip

echo "=================================================="
echo "Starting Inference..."
echo "Target Save Directory: $FINAL_SAVE_DIR"
echo "=================================================="
# Execute the python script with the configured arguments
python inference.py \
    --input_root "$INPUT_DIR" \
    --checkpoint "$CHECKPOINT_PATH" \
    --save_root "$FINAL_SAVE_DIR" \
    --gpu_list $GPU_LIST \
    --zip_name "$ZIP_NAME" \
    --whole_image

echo "=================================================="
echo "Inference Completed Successfully!"
echo "The submission zip is saved at: ${FINAL_SAVE_DIR}/${ZIP_NAME}"
echo "=================================================="