#!/bin/bash
##############################################################################
# NTIRE 2026 RAIM MEF Challenge - Inference Script
# Track 2: Multi-Exposure Image Fusion
# Team: NTR
##############################################################################

set -e

# Configuration
MODEL_PATH="${MODEL_PATH:-./checkpoints/raim_mef_phase2j/snapshot/net_final.pth}"
INPUT_DIR="${INPUT_DIR:-/home/notebook/data/personal/S9059954/HDR-Competition/datasets/test-stage2-wo-gt}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/test-stage2_results_NTIRE26_raim_mef_miketjc_final_submission/}"
STRIDE="${STRIDE:-128}"
ENSEMBLE="${ENSEMBLE:-True}"
TILE_SIZE="${TILE_SIZE:-512}"
JPEG_QUALITY="${JPEG_QUALITY:-95}"

# Check model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model checkpoint not found at $MODEL_PATH"
    echo "Please download and place it there"
    exit 1
fi

# Check input data exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input data not found at $INPUT_DIR"
    echo "Please download Phase 3 test data"
    exit 1
fi

echo "=========================================================================="
echo "NTIRE 2026 RAIM MEF Challenge - Inference"
echo "=========================================================================="
echo "Model:      $MODEL_PATH"
echo "Input:      $INPUT_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Stride:     $STRIDE"
echo "Ensemble:   $ENSEMBLE"
echo "=========================================================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference
python scripts/raim_mef/infer_mef.py \
  --model_path "$MODEL_PATH" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --stride "$STRIDE" \
  $([ "$ENSEMBLE" = "True" ] && echo "--ensemble") \
  --tile_size "$TILE_SIZE" \
  --num_exposures 5 \
  --in_channels 15 \
  --jpeg_quality "$JPEG_QUALITY" \
  --device cuda

echo ""
echo "=========================================================================="
echo "✓ Inference complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Files: $(ls $OUTPUT_DIR/*.jpg | wc -l) JPEG images"
echo "=========================================================================="
