#!/bin/bash
# Demo script for MLM training - smoke test with small dataset
# Author: ywangmu from HKUST

set -e

echo "========================================="
echo "BBencoder MLM Pre-training - Demo"
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

python3 train_encoder_mlm.py \
  --data dataset/final_training_data.pkl \
  --num-samples 100 \
  --epochs 5 \
  --batch-size 8 \
  --lr 1e-3 \
  --max-seq-len 256 \
  --embed-dim 64 \
  --hidden-dim 128 \
  --device cuda \
  --output-dir models_mlm_demo \
  --save-every 2 \
  --validation-split 0.2 \
  --early-stopping-patience 3 \
  --debug

echo ""
echo "========================================="
echo "Demo completed!"
echo "Check models_mlm_demo/ for saved models"
echo "========================================="

