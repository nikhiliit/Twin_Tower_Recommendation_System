#!/usr/bin/env bash
# scripts/serve.sh
# Launch the FastAPI serving API locally without Docker.
#
# Usage:
#   bash scripts/serve.sh                      # defaults
#   DEVICE=cuda bash scripts/serve.sh          # GPU serving
#   AB_TRAFFIC_SPLIT=0.8 bash scripts/serve.sh # 80% to variant A
#
# Environment variables (all optional, sensible defaults):
#   MODEL_A_CHECKPOINT  : path to model-A checkpoint (default: checkpoints/best_model.pt)
#   MODEL_B_CHECKPOINT  : path to model-B checkpoint (default: same as A)
#   FAISS_INDEX_PATH    : base path for FAISS index files (default: checkpoints/faiss_index)
#   PROCESSED_DIR       : path to data/processed/ (default: data/processed)
#   CONFIG_PATH         : path to YAML config (default: configs/hardneg_config.yaml)
#   AB_TRAFFIC_SPLIT    : float [0,1] — fraction to variant A (default: 0.5)
#   DEVICE              : cpu | cuda (default: cpu)
#   PORT                : API port (default: 8000)

set -euo pipefail

export MODEL_A_CHECKPOINT="${MODEL_A_CHECKPOINT:-checkpoints/best_model.pt}"
export MODEL_B_CHECKPOINT="${MODEL_B_CHECKPOINT:-$MODEL_A_CHECKPOINT}"
export FAISS_INDEX_PATH="${FAISS_INDEX_PATH:-checkpoints/faiss_index}"
export PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
export CONFIG_PATH="${CONFIG_PATH:-configs/hardneg_config.yaml}"
export AB_TRAFFIC_SPLIT="${AB_TRAFFIC_SPLIT:-0.5}"
export DEVICE="${DEVICE:-cpu}"
PORT="${PORT:-8000}"

echo "============================================="
echo " Two-Tower Retrieval — Serving API"
echo "============================================="
echo " Model A  : $MODEL_A_CHECKPOINT"
echo " Model B  : $MODEL_B_CHECKPOINT"
echo " FAISS    : $FAISS_INDEX_PATH"
echo " Split    : $AB_TRAFFIC_SPLIT (A) / $(python -c "print(round(1 - $AB_TRAFFIC_SPLIT, 2))") (B)"
echo " Device   : $DEVICE"
echo " Port     : $PORT"
echo "============================================="
echo ""

uvicorn src.serving.app:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info \
    --reload
