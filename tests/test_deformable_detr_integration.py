
import unittest
import torch
import torch.nn as nn
from models.deformable_detr import DeformableDETR
from util.misc import NestedTensor

class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.strides = [8, 16, 32]
        self.num_channels = [512, 1024, 2048]

    def forward(self, x):
        # Return features and pos
        # x is NestedTensor
        # features: list of NestedTensor
        # pos: list of Tensor
        batch_size = x.tensors.shape[0]
        features = []
        pos = []
        for i, c in enumerate(self.num_channels):
            feat = torch.randn(batch_size, c, 10, 10)
            mask = torch.zeros(batch_size, 10, 10, dtype=torch.bool)
            features.append(NestedTensor(feat, mask))
            pos.append(torch.randn(batch_size, 256, 10, 10)) # hidden_dim=256
        return features, pos
    
    def __getitem__(self, idx):
        # DeformableDETR calls self.backbone[1] to get pos encoding if extra levels
        if idx == 1:
            return PosEnco()
        return self

class PosEnco(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.randn(x.tensors.shape[0], 256, x.tensors.shape[2], x.tensors.shape[3])

class MockTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 256
        self.decoder = nn.Module()
        self.decoder.num_layers = 6

    def forward(self, srcs, masks, pos, query_embeds):
        bs = srcs[0].shape[0]
        num_queries = 300
        hs = torch.randn(6, bs, num_queries, 256)
        init_reference = torch.rand(bs, num_queries, 2)
        inter_references = torch.rand(6, bs, num_queries, 2)
        enc_outputs_class = torch.randn(bs, num_queries, 91)
        enc_outputs_coord_unact = torch.randn(bs, num_queries, 4)
        return hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact

class TestDeformableIntegration(unittest.TestCase):
    def test_forward(self):
        backbone = MockBackbone()
        transformer = MockTransformer()
        num_classes = 91
        num_queries = 300
        num_feature_levels = 4
        
        model = DeformableDETR(backbone, transformer, num_classes, num_queries, num_feature_levels)
        
        # Check if n_bins is set
        self.assertEqual(model.n_bins, 16)
        
        # Check bbox_embed shape
        # Last layer of first bbox_embed (if list) or checking single one
        if isinstance(model.bbox_embed, nn.ModuleList):
            last_linear = model.bbox_embed[0].layers[-1]
        else:
            last_linear = model.bbox_embed.layers[-1]
            
        self.assertEqual(last_linear.out_features, 4 * 16)
        
        # Test Forward
        images = torch.randn(2, 3, 200, 200)
        masks = torch.zeros(2, 200, 200, dtype=torch.bool)
        samples = NestedTensor(images, masks)
        
        out = model(samples)
        
        self.assertIn('pred_logits', out)
        self.assertIn('pred_boxes', out)
        self.assertIn('pred_entropy', out)
        self.assertIn('pred_box_dist_logits', out)
        
        # Check shapes
        self.assertEqual(out['pred_boxes'].shape, (2, 300, 4))
        self.assertEqual(out['pred_entropy'].shape, (2, 300))
        self.assertEqual(out['pred_box_dist_logits'].shape, (2, 300, 4 * 16))
        
        # Check range of boxes (should be roughly 0-1 if reference is valid)
        # Note: raw untrained output might be outside if deltas are large, but with init 0 deltas, it should be close to reference
        self.assertTrue(torch.isnan(out['pred_boxes']).sum() == 0)

if __name__ == '__main__':
    unittest.main()
