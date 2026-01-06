
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

# Ensure we can import modules
sys.path.append(os.getcwd())

from main import get_args_parser
from models import build_model
import util.misc as utils
from unittest.mock import patch
from tests.test_deformable_detr_integration import MockBackbone, MockTransformer

class TestFullChain(unittest.TestCase):
    @patch('models.deformable_detr.build_backbone')
    @patch('models.deformable_detr.build_deforamble_transformer')
    def test_build_and_forward(self, mock_build_transformer, mock_build_backbone):
        # Setup Mocks
        mock_build_backbone.return_value = MockBackbone()
        mock_build_transformer.return_value = MockTransformer() # Need to ensure dims match

        # 1. Parse Args
        parser = get_args_parser()
        # Simulate command line arguments
        cmd_args = [
            '--backbone', 'resnet50',
            '--num_queries', '300', # Match MockTransformer defaults
            '--batch_size', '2',
            '--dynamic_routing',
            '--thresh_cls', '0.5',
            '--thresh_uncertainty', '0.3',
            '--distill_loss_coef', '1.0',
            '--hidden_dim', '256'
        ]
        args = parser.parse_args(cmd_args)
        
        # Mocking device to cpu for test
        args.device = 'cpu'
        
        # 2. Build Model
        model, criterion, postprocessors = build_model(args)
        model.eval()
        
        # Verify Model Config
        # The model might be wrapped in DetrSegm if masks is on, but here it is not.
        if isinstance(model, nn.Module) and hasattr(model, 'dynamic_routing'):
             real_model = model
        elif hasattr(model, 'detr'):
             real_model = model.detr
        else:
             # It might be the DeformableDETR itself
             real_model = model
             
        self.assertTrue(real_model.dynamic_routing)
        self.assertEqual(real_model.thresh_cls, 0.5)
        
        # Verify Criterion Config
        self.assertIn('loss_distill', criterion.weight_dict)
        self.assertEqual(criterion.weight_dict['loss_distill'], 1.0)
        self.assertIn('distill', criterion.losses)
        
        # 3. Forward Pass (Fake Data)
        # NestedTensor input
        inputs = torch.randn(2, 3, 64, 64)
        masks = torch.zeros(2, 64, 64, dtype=torch.bool)
        from util.misc import NestedTensor
        samples = NestedTensor(inputs, masks)
        
        out = model(samples)
        
        self.assertIn('pred_logits', out)
        self.assertIn('pred_boxes', out)
        self.assertIn('pred_entropy', out)
        self.assertIn('pred_box_dist_logits', out)
        
        # 4. Loss Pass (Fake Targets)
        targets = []
        for _ in range(2):
            targets.append({
                'labels': torch.tensor([1, 2], dtype=torch.long),
                'boxes': torch.rand(2, 4) # cx, cy, w, h
            })
            
        losses = criterion(out, targets)
        
        self.assertIn('loss_distill', losses)
        print("Distillation Loss:", losses['loss_distill'].item())
        
        # 5. Check DR Stats
        self.assertIn('dr_stats', out)
        if hasattr(real_model, 'dynamic_routing') and real_model.dynamic_routing:
             stats = out['dr_stats']
             self.assertIn('flops_reduction', stats)
             self.assertIn('exit_counts', stats)
             print("DR Stats:", stats)

if __name__ == '__main__':
    unittest.main()
