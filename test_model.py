
import torch
import sys
sys.path.append('/ghome/wangyx3/Deformable_detr')
from models.deformable_detr import build

class Args:
    dataset_file = 'coco'
    device = 'cpu'
    num_queries = 100
    num_feature_levels = 4
    aux_loss = True
    with_box_refine = True
    two_stage = False
    masks = False
    hidden_dim = 256
    nheads = 8
    enc_layers = 6
    dec_layers = 6
    dim_feedforward = 1024
    dropout = 0.1
    dec_n_points = 4
    enc_n_points = 4
    cls_loss_coef = 1.0
    bbox_loss_coef = 5.0
    giou_loss_coef = 2.0
    mask_loss_coef = 1.0
    dice_loss_coef = 1.0
    focal_alpha = 0.25
    lr_backbone = 1e-5
    backbone = 'resnet50'
    dilation = False
    position_embedding = 'sine'
    
args = Args()
model, criterion, postprocessors = build(args)
print("Model built successfully")

dummy_image = torch.randn(1, 3, 800, 800)
mask = torch.zeros(1, 800, 800, dtype=torch.bool)
from util.misc import NestedTensor
samples = NestedTensor(dummy_image, mask)

output = model(samples)
print("Output keys:", output.keys())
print("Pred boxes shape:", output['pred_boxes'].shape)
print("Pred dist shape:", output['pred_dist'].shape)
print("Entropy shape:", output['pred_entropy'].shape)
print("Distill loss test...")

# Fake targets
targets = [{'labels': torch.tensor([1]), 'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
losses = criterion(output, targets)
print("Loss keys:", losses.keys())
