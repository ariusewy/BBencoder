#!/bin/bash
# BBencoder Supervised Training Script
# Author: ywangmu from HKUST
# Run with: bash train.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_log_supervised_${TIMESTAMP}.txt"

echo "================================================================================"
echo "BBencoder - Supervised Training with BB Similarity"
echo "================================================================================"
echo "Starting training at $(date)"
echo "Log file: ${LOG_FILE}"
echo ""

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nanogpt

echo "Training configuration:"
echo "  - Samples: 20000"
echo "  - Epochs: 100"
echo "  - Batch size: 64"
echo "  - Pairs per sample: 5"
echo "  - Dependency mode: ENABLED (window=3)"
echo "  - Model: embed=192, hidden=384, output=96"
echo ""

nohup python3 train_full_model_supervised.py \
    --num-samples 20000 \
    --epochs 100 \
    --batch-size 64 \
    --num-pairs 5 \
    --embed-dim 192 \
    --hidden-dim 384 \
    --output-dim 96 \
    --max-length 1600 \
    --device cuda \
    --validation-split 0.2 \
    --early-stopping-patience 20 \
    --enable-dep \
    --dep-window 3 \
    --beta-dep 0.5 \
    --gamma-mis 0.05 \
    > ${LOG_FILE} 2>&1 &

PID=$!
echo "Training started with PID: ${PID}"
echo ""
echo "Useful commands:"
echo "  Monitor progress:  tail -f ${LOG_FILE}"
echo "  Follow live:       tail -f ${LOG_FILE} | grep -E 'Epoch|BB Similarity|Best'"
echo "  Check status:      ps -p ${PID}"
echo "  Check GPU usage:   nvidia-smi"
echo "  Kill training:     kill ${PID}"
echo ""
echo "Training will take approximately 8-12 hours"
echo "Results will be saved to: models_supervised/"
echo "================================================================================"