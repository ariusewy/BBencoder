#!/bin/bash
# Two-stage training pipeline for BBencoder
# Stage 1: MLM pre-training
# Stage 2: Supervised fine-tuning with optional pretrained checkpoint
# Author: ywangmu from HKUST

set -euo pipefail

MODEL=${1:-lstm}
DATA_PATH=${DATA_PATH:-dataset/final_training_data.pkl}
DEVICE=${DEVICE:-cuda}

# Stage-specific hyperparameters (override via environment variables if needed)
MLM_SAMPLES=${MLM_SAMPLES:-50000}
MLM_EPOCHS=${MLM_EPOCHS:-50}
MLM_BATCH=${MLM_BATCH:-32}
MLM_LR=${MLM_LR:-1e-3}

SUP_SAMPLES=${SUP_SAMPLES:-20000}
SUP_EPOCHS=${SUP_EPOCHS:-100}
SUP_BATCH=${SUP_BATCH:-64}
SUP_LR=${SUP_LR:-5e-4}
SUP_OUTPUT_DIM=${SUP_OUTPUT_DIM:-96}
SUP_NUM_PAIRS=${SUP_NUM_PAIRS:-5}

EMBED_DIM=${EMBED_DIM:-192}
HIDDEN_DIM=${HIDDEN_DIM:-384}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1600}
DROPOUT=${DROPOUT:-0.1}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
STAGE1_DIR=${STAGE1_DIR:-models_stage1_${MODEL}_${TIMESTAMP}}
STAGE2_DIR=${STAGE2_DIR:-models_stage2_${MODEL}_${TIMESTAMP}}

echo "=========================================="
echo "Two-Stage Training Pipeline"
echo "Model:        ${MODEL}"
echo "Data:         ${DATA_PATH}"
echo "Device:       ${DEVICE}"
echo "Stage1 dir:   ${STAGE1_DIR}"
echo "Stage2 dir:   ${STAGE2_DIR}"
echo "=========================================="

echo ""
echo ">>> Stage 1: MLM Pre-training"
python3 train.py \
  --model "${MODEL}" \
  --task mlm \
  --mode train \
  --data "${DATA_PATH}" \
  --num-samples "${MLM_SAMPLES}" \
  --epochs "${MLM_EPOCHS}" \
  --batch-size "${MLM_BATCH}" \
  --lr "${MLM_LR}" \
  --embed-dim "${EMBED_DIM}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --dropout "${DROPOUT}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --device "${DEVICE}" \
  --output-dir "${STAGE1_DIR}"

PRETRAINED_CKPT="${STAGE1_DIR}/best_model.pt"

if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
  echo "ERROR: Stage 1 checkpoint not found at ${PRETRAINED_CKPT}"
  exit 1
fi

echo ""
echo ">>> Stage 2: Supervised Fine-tuning"
python3 train.py \
  --model "${MODEL}" \
  --task supervised \
  --mode train \
  --data "${DATA_PATH}" \
  --num-samples "${SUP_SAMPLES}" \
  --epochs "${SUP_EPOCHS}" \
  --batch-size "${SUP_BATCH}" \
  --lr "${SUP_LR}" \
  --embed-dim "${EMBED_DIM}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --output-dim "${SUP_OUTPUT_DIM}" \
  --dropout "${DROPOUT}" \
  --max-seq-len "${MAX_SEQ_LEN}" \
  --num-pairs "${SUP_NUM_PAIRS}" \
  --enable-dep \
  --device "${DEVICE}" \
  --pretrained-checkpoint "${PRETRAINED_CKPT}" \
  --output-dir "${STAGE2_DIR}"

echo ""
echo "=========================================="
echo "Two-Stage Training Complete!"
echo "Stage 1 output: ${STAGE1_DIR}"
echo "Stage 2 output: ${STAGE2_DIR}"
echo "Final model:    ${STAGE2_DIR}/best_model.pt"
echo "=========================================="



