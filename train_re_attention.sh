#!/usr/bin/env bash

# Set the path to your COCO dataset
COCO_PATH=/path/to/your/coco

# Set the output directory
OUTPUT_DIR=exps/re_attention_only

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run training
# Note: 
# 1. We use --train_re_attention_only to freeze other parameters.
# 2. You might want to load a pretrained model using --resume or --frozen_weights if you have one,
#    otherwise the backbone and transformer will be random initialized (which is bad for re-attention training).
#    Assuming you want to fine-tune on top of a trained Deformable DETR.
#    Here I add a placeholder for --resume. Please replace it with your checkpoint path.

# Example checkpoint path (replace with actual path)
# CHECKPOINT=path/to/r50_deformable_detr-checkpoint.pth

python -u main.py \
    --output_dir ${OUTPUT_DIR} \
    --coco_path ${COCO_PATH} \
    --train_re_attention_only \
    --resume ${CHECKPOINT:-""} \
    --epochs 50 \
    --lr 2e-4 \
    --batch_size 2 \
    --num_workers 4 \
    --with_box_refine \
    --two_stage \
    "$@"
