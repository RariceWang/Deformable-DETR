#!/usr/bin/env bash

# Set the path to your COCO dataset
COCO_PATH=/workspace/data/coco/

# Set the output directory
OUTPUT_DIR=/output/deformable_detr_re_attention/
# Create output directory
mkdir -p ${OUTPUT_DIR}

# Set TORCH_HOME to a writable directory
export TORCH_HOME=$(pwd)/.cache
mkdir -p ${TORCH_HOME}

# Training config for 8x 2080 Ti
GPUS=4
BATCH_SIZE=2         # per-GPU batch; adjust if memory allows
NUM_WORKERS=4

# Example checkpoint path (replace with actual path)
CHECKPOINT=/workspace/checkpoint.pth

# Run training in distributed mode
bash ./tools/run_dist_launch.sh ${GPUS} python -u main.py \
    --output_dir ${OUTPUT_DIR} \
    --coco_path ${COCO_PATH} \
    --resume ${CHECKPOINT:-""} \
    --epochs 10 \
    --lr 2e-4 \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --with_box_refine \
    "$@"
