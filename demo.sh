#!/bin/bash
# Demo script for BBencoder unified training
# Author: ywangmu from HKUST
# Quick smoke test with debug mode

set -e

echo "=========================================="
echo "BBencoder Demo - Quick Smoke Test"
echo "=========================================="

# Default settings
MODEL=${1:-lstm}          # lstm, babygpt, gpt2-small, gpt2
TASK=${2:-mlm}            # mlm, supervised, supervised-finetune, two-stage
PRETRAINED_CKPT=${3:-}    # used for supervised-finetune

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

run_demo () {
  local MODEL_NAME=$1
  local TASK_NAME=$2
  shift 2
  python3 train.py \
    --model "$MODEL_NAME" \
    --task "$TASK_NAME" \
    --mode demo \
    --num-samples 20 \
    --epochs 3 \
    --batch-size 4 \
    --lr 1e-3 \
    --max-seq-len 128 \
    --embed-dim 64 \
    --hidden-dim 128 \
    --output-dim 32 \
    --dropout 0.1 \
    --device cuda \
    --validation-split 0.2 \
    --save-every 1 \
    --early-stopping-patience 0 \
    --seed 42 \
    "$@"
}

case "$TASK" in
  supervised-finetune)
    if [[ -z "$PRETRAINED_CKPT" ]]; then
      echo "Error: supervised-finetune requires a pretrained checkpoint path as the third argument."
      exit 1
    fi
    echo "Model: $MODEL"
    echo "Task: supervised (fine-tune)"
    echo "Pretrained checkpoint: $PRETRAINED_CKPT"
    echo ""
    run_demo "$MODEL" "supervised" --pretrained-checkpoint "$PRETRAINED_CKPT"
    ;;

  two-stage)
    STAGE1_OUT="demo_stage1_${MODEL}_${TIMESTAMP}"
    STAGE2_OUT="demo_stage2_${MODEL}_${TIMESTAMP}"
    echo "Model: $MODEL"
    echo "Task: two-stage (MLM -> Supervised)"
    echo "Stage1 output: $STAGE1_OUT"
    echo "Stage2 output: $STAGE2_OUT"
    echo ""

    echo "[Stage 1] MLM Demo..."
    run_demo "$MODEL" "mlm" --output-dir "$STAGE1_OUT"

    PRETRAINED="${STAGE1_OUT}/best_model.pt"
    if [[ ! -f "$PRETRAINED" ]]; then
      echo "Error: Stage 1 checkpoint not found at $PRETRAINED"
      exit 1
    fi

    echo ""
    echo "[Stage 2] Supervised Demo with pretrained weights..."
    run_demo "$MODEL" "supervised" \
      --pretrained-checkpoint "$PRETRAINED" \
      --output-dir "$STAGE2_OUT"

    echo ""
    echo "Two-stage demo complete!"
    echo "  Stage 1 best model: $PRETRAINED"
    echo "  Stage 2 best model: ${STAGE2_OUT}/best_model.pt"
    ;;

  *)
    echo "Model: $MODEL"
    echo "Task: $TASK"
    echo ""
    run_demo "$MODEL" "$TASK"
    ;;
esac

echo ""
echo "=========================================="
echo "Demo Complete!"
echo "=========================================="

