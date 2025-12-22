#!/bin/bash
# Unified training script for BBencoder (full training runs)
# Author: ywangmu from HKUST
# Usage:
#   bash train.sh lstm mlm
#   bash train.sh lstm supervised
#   bash train.sh lstm supervised-finetune /path/to/best_model.pt
#   bash train.sh lstm two-stage

set -euo pipefail

MODEL=${1:-lstm}
TASK=${2:-mlm}            # mlm | supervised | supervised-finetune | two-stage
PRETRAINED_CKPT=${3:-}    # used for supervised-finetune

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_FILE:-${LOG_DIR}/train_${MODEL}_${TASK}_${TIMESTAMP}.log}
echo "Logging to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
DATA_PATH=${DATA_PATH:-dataset/final_training_data_fp_fixed.pkl}
DEVICE=${DEVICE:-cuda}
VAL_SPLIT=${VAL_SPLIT:-0.2}
SAVE_EVERY=${SAVE_EVERY:-10}
EARLY_STOP=${EARLY_STOP:-10}
LR_SCHEDULER=${LR_SCHEDULER:-none}          # none | plateau
LR_DECAY_FACTOR=${LR_DECAY_FACTOR:-0.5}     # factor for ReduceLROnPlateau
LR_DECAY_PATIENCE=${LR_DECAY_PATIENCE:-5}   # epochs with no val improvement before LR decay
MIN_LR=${MIN_LR:-1e-6}                      # lower bound for LR when using scheduler
PRECOMPUTED_PAIRS=${PRECOMPUTED_PAIRS:-}

# MLM defaults
MLM_SAMPLES=${MLM_SAMPLES:-}
MLM_EPOCHS=${MLM_EPOCHS:-50}
MLM_BATCH=${MLM_BATCH:-32}
MLM_LR=${MLM_LR:-1e-3}
# Keep MLM and Supervised stages aligned on sequence length for checkpoint reuse
MLM_MAX_SEQ_LEN=${MLM_MAX_SEQ_LEN:-1600}
MLM_EMBED_DIM=${MLM_EMBED_DIM:-192}
MLM_HIDDEN_DIM=${MLM_HIDDEN_DIM:-384}
MLM_DROPOUT=${MLM_DROPOUT:-0.1}

# Supervised defaults
SUP_SAMPLES=${SUP_SAMPLES:-}
SUP_EPOCHS=${SUP_EPOCHS:-100}
SUP_BATCH=${SUP_BATCH:-16}
SUP_LR=${SUP_LR:-5e-4}
SUP_MAX_SEQ_LEN=${SUP_MAX_SEQ_LEN:-1600}
SUP_EMBED_DIM=${SUP_EMBED_DIM:-192}
SUP_HIDDEN_DIM=${SUP_HIDDEN_DIM:-384}
SUP_OUTPUT_DIM=${SUP_OUTPUT_DIM:-96}
SUP_NUM_PAIRS=${SUP_NUM_PAIRS:-5}
SUP_ENABLE_DEP=${SUP_ENABLE_DEP:-1}
SUP_DEP_WINDOW=${SUP_DEP_WINDOW:-5}
SUP_BETA_DEP=${SUP_BETA_DEP:-0.5}
SUP_GAMMA_MIS=${SUP_GAMMA_MIS:-0.05}

run_train () {
  local MODEL_NAME=$1
  local TASK_NAME=$2
  shift 2
  python3 train.py \
    --model "$MODEL_NAME" \
    --task "$TASK_NAME" \
    --mode train \
    --data "$DATA_PATH" \
    --device "$DEVICE" \
    --validation-split "$VAL_SPLIT" \
    --save-every "$SAVE_EVERY" \
    --early-stopping-patience "$EARLY_STOP" \
    --lr-scheduler "$LR_SCHEDULER" \
    --lr-decay-factor "$LR_DECAY_FACTOR" \
    --lr-decay-patience "$LR_DECAY_PATIENCE" \
    --min-lr "$MIN_LR" \
    "$@"
}

run_mlm () {
  local EXTRA_ARGS=()
  if [[ -n "${MLM_SAMPLES}" ]]; then
    EXTRA_ARGS+=(--num-samples "$MLM_SAMPLES")
  fi
  run_train "$MODEL" "mlm" \
    "${EXTRA_ARGS[@]}" \
    --epochs "$MLM_EPOCHS" \
    --batch-size "$MLM_BATCH" \
    --lr "$MLM_LR" \
    --max-seq-len "$MLM_MAX_SEQ_LEN" \
    --embed-dim "$MLM_EMBED_DIM" \
    --hidden-dim "$MLM_HIDDEN_DIM" \
    --dropout "$MLM_DROPOUT" \
    "$@"
}

run_supervised () {
  local EXTRA_ARGS=()
  if [[ "${SUP_ENABLE_DEP}" == "1" ]]; then
    EXTRA_ARGS+=(--enable-dep --dep-window "$SUP_DEP_WINDOW" --beta-dep "$SUP_BETA_DEP" --gamma-mis "$SUP_GAMMA_MIS")
  fi
  if [[ -n "${SUP_SAMPLES}" ]]; then
    EXTRA_ARGS+=(--num-samples "$SUP_SAMPLES")
  fi
  if [[ -n "${PRECOMPUTED_PAIRS}" ]]; then
    EXTRA_ARGS+=(--precomputed-pairs "$PRECOMPUTED_PAIRS")
  fi
  run_train "$MODEL" "supervised" \
    "${EXTRA_ARGS[@]}" \
    --epochs "$SUP_EPOCHS" \
    --batch-size "$SUP_BATCH" \
    --lr "$SUP_LR" \
    --max-seq-len "$SUP_MAX_SEQ_LEN" \
    --embed-dim "$SUP_EMBED_DIM" \
    --hidden-dim "$SUP_HIDDEN_DIM" \
    --output-dim "$SUP_OUTPUT_DIM" \
    --num-pairs "$SUP_NUM_PAIRS" \
    "$@"
}

echo "================================================================================"
echo "BBencoder - Full Training"
echo "================================================================================"
echo "Model:       $MODEL"
echo "Task:        $TASK"
echo "Data:        $DATA_PATH"
echo "Device:      $DEVICE"
echo "Timestamp:   $TIMESTAMP"
echo "================================================================================"

case "$TASK" in
  supervised-finetune)
    if [[ -z "$PRETRAINED_CKPT" ]]; then
      echo "Error: supervised-finetune requires a pretrained checkpoint path as the third argument."
      exit 1
    fi
    echo "Pretrained checkpoint: $PRETRAINED_CKPT"
    run_supervised --pretrained-checkpoint "$PRETRAINED_CKPT"
    ;;

  two-stage)
    STAGE1_OUT="train_stage1_${MODEL}_${TIMESTAMP}"
    STAGE2_OUT="train_stage2_${MODEL}_${TIMESTAMP}"
    echo "Running two-stage training."
    echo "Stage1 output dir: $STAGE1_OUT"
    echo "Stage2 output dir: $STAGE2_OUT"
    echo ""

    echo "[Stage 1] MLM training..."
    run_mlm --output-dir "$STAGE1_OUT"

    PRETRAINED="${STAGE1_OUT}/best_model.pt"
    if [[ ! -f "$PRETRAINED" ]]; then
      echo "Error: Stage 1 checkpoint not found at $PRETRAINED"
      exit 1
    fi

    echo ""
    echo "[Stage 2] Supervised fine-tuning..."
    # For two-stage training, always generate pairs on-the-fly with the current tokenizer
    # and ignore any PRECOMPUTED_PAIRS environment setting.
    PRECOMPUTED_PAIRS= run_supervised --pretrained-checkpoint "$PRETRAINED" --output-dir "$STAGE2_OUT"

    echo ""
    echo "Two-stage training complete."
    echo "  Stage 1 best model: $PRETRAINED"
    echo "  Stage 2 best model: ${STAGE2_OUT}/best_model.pt"
    ;;

  mlm)
    run_mlm
    ;;

  supervised)
    run_supervised
    ;;

  *)
    echo "Unsupported task: $TASK"
    exit 1
    ;;
esac

echo "================================================================================"
echo "Training pipeline finished."
echo "================================================================================"