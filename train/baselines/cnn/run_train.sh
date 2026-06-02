#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python train.py \
    --train_json  ../../data_gen/trainset_sharegpt_data_may23.json \
    --test_json   ../../data_gen/testset_sharegpt_data_may23.json \
    --epochs      50 \
    --batch_size  32 \
    --lr          1e-4 \
    --save_path   checkpoints/best_model.pth \
    --plots_dir   plots \
    --log_path    logs/training.txt
