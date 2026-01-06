
import unittest
import torch
from models.deformable_detr import DeformableDETR
from tests.test_deformable_detr_integration import MockBackbone, MockTransformer

class TestDynamicRouting(unittest.TestCase):
    def test_routing_logic(self):
        # We need to access the 'out' and check if it used early exit.
        # How to verify?
        # We can set thresholds such that Layer 0 exits immediately.
        # Then the final output should match Layer 0 output.
        
        backbone = MockBackbone()
        transformer = MockTransformer()
        model = DeformableDETR(backbone, transformer, 91, 300, 4)
        
        # Enable Routing
        model.dynamic_routing = True
        model.thresh_cls = 0.0 # Everything passes score
        model.thresh_uncertainty = 100.0 # Everything passes uncertainty (low entropy < high thresh)
        # So everything should exit at Layer 0 (first layer).
        
        # But wait, layer 0 is usually object queries.
        # The loop iterates `range(hs.shape[0])`.
        # Layer 0 output is the first one.
        
        inputs = torch.randn(2, 3, 200, 200)
        from util.misc import NestedTensor
        samples = NestedTensor(inputs, torch.zeros(2, 200, 200, dtype=torch.bool))
        
        # We need to capture the intermediate outputs to verify.
        # The model returns the 'out' dict which usually contains the *last* layer.
        # With checking implemented, 'pred_boxes' should be composed of mixed layers.
        
        # If we force early exit at Layer 0, the final output should be identical to Layer 0 output.
        # Layer 0 output is in `aux_outputs[0]`? No.
        # The model logic I need to implement will store the mixed result in `pred_boxes`.
        
        # Let's run forward.
        out = model(samples)
        
        # Inspect outputs
        # To strictly verify, I might need to inspect internals or mock the heads to produce distinct values per layer.
        pass

if __name__ == '__main__':
    unittest.main()
