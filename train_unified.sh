#!/bin/bash
# Full training script for BBencoder
# Author: ywangmu from HKUST

set -e

echo "=========================================="
echo "BBencoder Full Training"
echo "=========================================="

# Parse arguments
MODEL=${1:-lstm}          # lstm, babygpt, gpt2-small, gpt2
TASK=${2:-mlm}            # mlm, supervised

echo "Model: $MODEL"
echo "Task: $TASK"
echo ""

# Task-specific parameters
if [ "$TASK" = "mlm" ]; then
    echo "MLM Task Configuration:"
    echo "  - Token masking: 15%"
    echo "  - Instruction masking: 15%"
    echo ""
    
    python3 train.py \
      --model $MODEL \
      --task mlm \
      --mode train \
      --num-samples 5000 \
      --epochs 50 \
      --batch-size 32 \
      --lr 1e-3 \
      --max-seq-len 512 \
      --embed-dim 64 \
      --hidden-dim 128 \
      --dropout 0.1 \
      --token-mask-prob 0.15 \
      --instr-mask-prob 0.15 \
      --device cuda \
      --validation-split 0.2 \
      --save-every 10 \
      --early-stopping-patience 10 \
      --grad-clip 1.0 \
      --seed 42

elif [ "$TASK" = "supervised" ]; then
    echo "Supervised Task Configuration:"
    echo "  - BB similarity based (Weighted LCS + Execution Unit Aware)"
    echo "  - Samples: 20000"
    echo "  - Epochs: 100"
    echo "  - Batch size: 64"
    echo "  - Pairs per sample: 5"
    echo "  - Dependency mode: ENABLED (window=3, beta=0.5, gamma=0.05)"
    echo "  - Model: embed=192, hidden=384, output=96"
    echo ""
    
    python3 train.py \
      --model $MODEL \
      --task supervised \
      --mode train \
      --num-samples 20000 \
      --epochs 100 \
      --batch-size 64 \
      --lr 1e-3 \
      --max-seq-len 1600 \
      --embed-dim 192 \
      --hidden-dim 384 \
      --output-dim 96 \
      --dropout 0.2 \
      --num-pairs 5 \
      --normalize maxlen \
      --enable-dep \
      --beta-dep 0.5 \
      --gamma-mis 0.05 \
      --dep-window 3 \
      --device cuda \
      --validation-split 0.2 \
      --save-every 10 \
      --early-stopping-patience 20 \
      --grad-clip 1.0 \
      --seed 42

else
    echo "Error: Unknown task '$TASK'"
    echo "Usage: $0 [model] [task]"
    echo "  model: lstm, babygpt, gpt2-small, gpt2"
    echo "  task: mlm, supervised"
    exit 1
fi

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="

