#!/bin/bash
##############################################################################
# NTIRE 2026 RAIM MEF Challenge - Training Script
# Track 2: Multi-Exposure Image Fusion
# Team: miketjc
##############################################################################

set -e

# Configuration
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./dataset/TrainSet_MEF/}"
VAL_DATA_PATH="${VAL_DATA_PATH:-./dataset/TestSet_MEF/}"
SAVE_FOLDER="${SAVE_FOLDER:-./checkpoints/raim_mef_phase3/}"
RESUME_FROM="${RESUME_FROM:-./checkpoints/raim_mef_phase2c/snapshot/net_final.pth}"

# Check data paths exist
if [ ! -d "$TRAIN_DATA_PATH" ]; then
    echo "Error: Training data not found at $TRAIN_DATA_PATH"
    echo "Please download TrainSet_MEF and extract it there"
    exit 1
fi

if [ ! -d "$VAL_DATA_PATH" ]; then
    echo "Error: Validation data not found at $VAL_DATA_PATH"
    echo "Please download TestSet_MEF and extract it there"
    exit 1
fi

echo "=========================================================================="
echo "NTIRE 2026 RAIM MEF Challenge - Training Phase 3"
echo "=========================================================================="
echo "Train data: $TRAIN_DATA_PATH"
echo "Val data:   $VAL_DATA_PATH"
echo "Save to:    $SAVE_FOLDER"
echo "Resume:     $RESUME_FROM"
echo "=========================================================================="

# Create output directory
mkdir -p "$SAVE_FOLDER"

# Run training
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --train_data_path "$TRAIN_DATA_PATH" \
  --val_data_path "$VAL_DATA_PATH" \
  --save_folder "$SAVE_FOLDER" \
  --net_name "TimeDiffiT_ResNet_color_128" \
  --in_channels 15 \
  --num_exposures 5 \
  --patch_size 256 \
  --batch_size 4 \
  --num_epochs 300 \
  --learning_rate 1e-5 \
  --loss_type "composite" \
  --lambda_lpips 0.8 \
  --resume_from "$RESUME_FROM" \
  --device cuda

echo ""
echo "=========================================================================="
echo "✓ Training complete!"
echo "Best model saved to: $SAVE_FOLDER/snapshot/net_best.pth"
echo "=========================================================================="
