#!/bin/bash
# Demo script for transformer models - quick test with different architectures
# Author: ywangmu from HKUST

set -e

echo "========================================="
echo "BBencoder Transformer Models - Demo"
echo "========================================="

# Activate conda environment if needed
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate nanogpt || { echo "Failed to activate conda env nanogpt"; exit 1; }
fi

# Set environment
export PYTHONPATH=/home/ywangmu/Proj:/home/ywangmu/Proj/BBencoder:/home/ywangmu/Proj/BBencoder/scripts:$PYTHONPATH
export MPLBACKEND=Agg

cd /home/ywangmu/Proj/BBencoder

echo ""
echo "========================================="
echo "Testing BabyGPT (small transformer)"
echo "========================================="

python trainers/train_mlm.py \
  --model babygpt \
  --data dataset/final_training_data.pkl \
  --num-samples 50 \
  --epochs 3 \
  --batch-size 4 \
  --lr 3e-4 \
  --max-seq-len 256 \
  --dropout 0.1 \
  --device cuda \
  --output-dir models_babygpt_demo \
  --save-every 1 \
  --validation-split 0.2 \
  --early-stopping-patience 2

echo ""
echo "========================================="
echo "Demo completed!"
echo "Check models_babygpt_demo/ for saved models"
echo "========================================="

