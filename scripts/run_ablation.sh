#!/bin/bash
# Run all 3 loss function experiments sequentially.
# Logs each to MLflow under the same experiment name.
# Prints comparison table at the end.

set -euo pipefail

echo "============================================="
echo " Two-Tower Retrieval Ablation Study"
echo "============================================="
echo ""

# Experiment 1: BPR Loss
echo "[1/3] Running BPR experiment..."
python scripts/train.py --config configs/bpr_config.yaml
echo ""

# Experiment 2: In-Batch Softmax Loss
echo "[2/3] Running In-Batch Softmax experiment..."
python scripts/train.py --config configs/inbatch_config.yaml
echo ""

# Experiment 3: Hard Negative Loss
echo "[3/3] Running Hard Negative experiment..."
python scripts/train.py --config configs/hardneg_config.yaml
echo ""

echo "============================================="
echo " All experiments complete!"
echo " View results in MLflow:"
echo "   mlflow ui --backend-store-uri mlruns/"
echo "============================================="
