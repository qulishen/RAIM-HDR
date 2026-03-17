#!/bin/bash
# ==============================================================================
# One-click 4-stage training script — NTIRE 2026 RAIM Track2
# Team: whu-vip
# ==============================================================================

# 自动跳转到训练核心目录 (Train_accelerate) 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../Train_accelerate"

# ==============================================================================
# USER CONFIGURATION — modify as needed
# ==============================================================================
# 指定使用的 GPU 编号，使用逗号分隔 (默认: "0,1,2,3"；如需单卡或指定卡可改为 "0" 或 "2,3")
GPU_IDS="0,1,2,3"
MASTER_PORT=29501
CONDA_ENV="whu_vip"

# Stages to run: set to 0 to skip, 1 to run
RUN_STAGE1=1
RUN_STAGE2=1
RUN_STAGE3=1
RUN_STAGE4=1

CONFIG_DIR="./configs"
LOG_ROOT="./checkpoints"
# ==============================================================================

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# 自动解析 GPU 数量并隔离环境可见的显卡
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
IFS=',' read -r -a gpu_array <<< "$GPU_IDS"
NUM_GPUS=${#gpu_array[@]}

echo "======================================================"
echo "  NTIRE 2026 RAIM Track2 — 4-Stage Training"
echo "  GPU IDs     : $GPU_IDS (Total: $NUM_GPUS)"
echo "  Master port : $MASTER_PORT"
echo "  Conda env   : $CONDA_ENV"
echo "  Log root    : $LOG_ROOT"
echo "======================================================"

run_stage() {
    local stage_id=$1
    local config=$2

    echo ""
    echo "------------------------------------------------------"
    echo "  Stage $stage_id — config: $config"
    echo "------------------------------------------------------"

    if [ "$NUM_GPUS" -gt 1 ]; then
        accelerate launch \
            --multi_gpu \
            --num_processes "$NUM_GPUS" \
            --main_process_port "$MASTER_PORT" \
            --mixed_precision fp16 \
            train_core.py --config "$config"
    else
        accelerate launch \
            --num_processes 1 \
            --mixed_precision fp16 \
            train_core.py --config "$config"
    fi

    echo "  Stage $stage_id finished."
}

[ "$RUN_STAGE1" -eq 1 ] && run_stage 1 "$CONFIG_DIR/AFUNet_GM_SFT_stage1.yaml"
[ "$RUN_STAGE2" -eq 1 ] && run_stage 2 "$CONFIG_DIR/AFUNet_GM_SFT_stage2.yaml"
[ "$RUN_STAGE3" -eq 1 ] && run_stage 3 "$CONFIG_DIR/AFUNet_GM_SFT_stage3.yaml"
[ "$RUN_STAGE4" -eq 1 ] && run_stage 4 "$CONFIG_DIR/AFUNet_GM_SFT_stage4.yaml"

echo ""
echo "======================================================"
echo "  All stages completed."
echo "  Final weights: $LOG_ROOT/AFUNet_GM_SFT_stage4/latest_ema_model.pth"
echo "======================================================"