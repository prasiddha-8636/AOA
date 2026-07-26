#!/bin/bash
# Run all experiments sequentially
# Usage: bash experiments/run_all.sh

METHODS=("learned" "sinusoidal" "rope" "alibi" "nope" "kerple" "cable")
GPU=${1:-0}

export CUDA_VISIBLE_DEVICES=$GPU

for method in "${METHODS[@]}"; do
    echo "========================================"
    echo "Training: $method"
    echo "========================================"
    python -m src.train --method "$method"
done

echo "========================================"
echo "Position Interpolation fine-tuning"
echo "========================================"
python -m src.train_pi \
    --pretrained checkpoints/rope/final.pt \
    --target_len 2048

echo "========================================"
echo "Evaluation"
echo "========================================"
for method in "${METHODS[@]}"; do
    echo "Evaluating: $method"
    python -m src.evaluate --method "$method" --checkpoint "checkpoints/${method}/best.pt"
done

echo "========================================"
echo "Profiling"
echo "========================================"
python -m src.profile

echo "All experiments complete."
