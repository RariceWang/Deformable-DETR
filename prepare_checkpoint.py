import torch
import os

input_path = '/ghome/wangyx3/Deformable_detr/checkpoint.pth'
output_path = '/ghome/wangyx3/Deformable_detr/checkpoint_no_optim.pth'

print("Loading {}...".format(input_path))
checkpoint = torch.load(input_path, map_location='cpu')

keys_to_remove = ['optimizer', 'lr_scheduler', 'epoch']
for key in keys_to_remove:
    if key in checkpoint:
        print("Removing {}...".format(key))
        del checkpoint[key]

print("Saving to {}...".format(output_path))
torch.save(checkpoint, output_path)
print("Done.")
