#!/usr/bin/env bash
export TORCH_HOME=/workspace/.cache/torch
export DETR_HOME=/workspace/.cache/detr
export PYTHONPATH=$PYTHONPATH:/workspace:/workspace/models/ops
# 解决显存碎片化问题
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py \
    --coco_path /workspace/data/coco/ \
    --resume /workspace/checkpoint/checkpoint.pth \
    --output_dir /output/Deformable-DETR/ \
    --batch_size 8 \
    --num_workers 4 \
    --backbone resnet50 \
    --distill_loss_coef 1.0
