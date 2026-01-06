
import unittest
import torch
import torch.nn as nn
from models.deformable_detr import SetCriterion

class MockMatcher(nn.Module):
    def forward(self, outputs, targets):
        # Return dummy indices
        # List of (src_idx, tgt_idx)
        return [[(torch.tensor([0]), torch.tensor([0]))]]

class TestCriterionDistill(unittest.TestCase):
    def test_distill_loss(self):
        matcher = MockMatcher()
        weight_dict = {'loss_distill': 1.0}
        losses = ['distill']
        
        criterion = SetCriterion(91, matcher, weight_dict, losses)
        
        # Construct outputs
        B, N, bins = 1, 1, 4
        teacher_logits = torch.randn(B, N, 4 * bins)
        student_logits = torch.randn(B, N, 4 * bins)
        
        # outputs dict structure
        # Must have 'pred_box_dist_logits' and 'aux_outputs' with correct depth
        
        aux_outputs = []
        for i in range(3): # 0, 1, 2
            layer_out = {'pred_box_dist_logits': torch.randn(B, N, 4*bins)}
            if i == 2:
                layer_out['pred_box_dist_logits'] = student_logits
            aux_outputs.append(layer_out)
            
        outputs = {
            'pred_logits': torch.zeros(B, N, 92),
            'pred_boxes': torch.zeros(B, N, 4),
            'pred_box_dist_logits': teacher_logits,
            'aux_outputs': aux_outputs
        }
        
        targets = [{'labels': torch.tensor([1]), 'boxes': torch.tensor([[0.5, 0.5, 0.5, 0.5]])}]
        
        loss_dict = criterion(outputs, targets)
        
        self.assertIn('loss_distill', loss_dict)
        self.assertTrue(loss_dict['loss_distill'] > 0)
        
        # Test if gradients propagate to student (aux_outputs[2])
        # student_logits requires grad?
        student_logits.requires_grad = True
        aux_outputs[2]['pred_box_dist_logits'] = student_logits 
        # Re-run
        loss_dict = criterion(outputs, targets)
        loss_dict['loss_distill'].backward()
        self.assertIsNotNone(student_logits.grad)

if __name__ == '__main__':
    unittest.main()
