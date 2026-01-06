
import unittest
import torch
import torch.nn as nn
from models.deformable_detr import DeformableDETR
from tests.test_deformable_detr_integration import MockBackbone, MockTransformer
from util.misc import NestedTensor

class TestDynamicRouting(unittest.TestCase):
    def test_routing_runs(self):
        backbone = MockBackbone()
        transformer = MockTransformer()
        model = DeformableDETR(backbone, transformer, 91, 300, 4)
        
        model.dynamic_routing = True
        model.thresh_cls = 0.0 # Everything exits
        model.thresh_uncertainty = 100.0 
        
        inputs = torch.randn(2, 3, 200, 200)
        masks = torch.zeros(2, 200, 200, dtype=torch.bool)
        samples = NestedTensor(inputs, masks)
        
        out = model(samples)
        
        self.assertIn('pred_logits', out)
        self.assertIn('pred_boxes', out)
        
        # Check shapes
        self.assertEqual(out['pred_boxes'].shape, (2, 300, 4))

if __name__ == '__main__':
    unittest.main()
