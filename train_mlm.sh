#!/bin/bash
# Full MLM training script for BBencoder
# Author: ywangmu from HKUST

set -e

echo "========================================="
echo "BBencoder MLM Pre-training - Full"
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
  --num-samples 5000 \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-3 \
  --max-seq-len 1600 \
  --embed-dim 192 \
  --hidden-dim 384 \
  --device cuda \
  --output-dir models_mlm \
  --save-every 10 \
  --validation-split 0.2 \
  --early-stopping-patience 10

echo ""
echo "========================================="
echo "Training completed!"
echo "Check models_mlm/ for saved models"
echo "========================================="

