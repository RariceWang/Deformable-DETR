import torch
import sys

ckpt_path = '/ghome/wangyx3/Deformable_detr/checkpoint.pth'
try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f"Keys: {ckpt.keys()}")
    if 'optimizer' in ckpt:
        print("Optimizer present")
    if 'model' in ckpt:
        print("Model present")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
