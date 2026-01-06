
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistributionHead(nn.Module):
    def __init__(self, hidden_dim, n_bins=16):
        super().__init__()
        self.n_bins = n_bins
        # We assume the output range is [0, 1] for relative coordinates l,t,r,b
        # or we learn a scale. D-Fine typically uses a reg_max. 
        # For simplicity in this test, let's assume bins are uniformly distributed in [0, 1]
        self.linear = nn.Linear(hidden_dim, 4 * n_bins)
        
        # Bins: 0, 1/(N-1), ..., 1? Or 0 to reg_max?
        # Let's assume standard DFL (Distribution Focal Loss) setup: values 0, 1, ..., n_bins-1
        # Then we scale the result.
        # But wait, D-Fine says "discrete distribution logits".
        
        # We will register a buffer for bin values
        # Let's assume prediction is in [0, 1] (normalized image coords)
        # So bins are linspace(0, 1, n_bins)
        
        self.register_buffer('bins', torch.linspace(0, 1, n_bins))

    def forward(self, x):
        # x: [B, N, hidden_dim]
        B, N, C = x.shape
        logits = self.linear(x) # [B, N, 4 * n_bins]
        logits = logits.view(B, N, 4, self.n_bins)
        
        probabilities = F.softmax(logits, dim=-1) # [B, N, 4, n_bins]
        
        # Integral expectation
        # sum(P(x) * x)
        # We broadcast bins to [B, N, 4, n_bins]
        expected_values = torch.sum(probabilities * self.bins, dim=-1) # [B, N, 4]
        
        # Entropy
        # H(P) = - sum P * log(P)
        # Using log_softmax for numerical stability might be better, but softmax + log is fine for logical test
        eps = 1e-8
        entropy = -torch.sum(probabilities * torch.log(probabilities + eps), dim=-1) # [B, N, 4]
        
        # Average entropy across 4 coordinates? Or max?
        # The dynamic routing usually uses a single scholar for the box. Sum or Mean.
        avg_entropy = entropy.mean(dim=-1) # [B, N]
        
        return expected_values, avg_entropy, probabilities

class TestDistributionHead(unittest.TestCase):
    def test_shape_and_range(self):
        hidden_dim = 256
        n_bins = 16
        num_queries = 10
        batch_size = 2
        
        head = DistributionHead(hidden_dim, n_bins)
        x = torch.randn(batch_size, num_queries, hidden_dim)
        
        coords, entropy, probs = head(x)
        
        # Check shapes
        self.assertEqual(coords.shape, (batch_size, num_queries, 4))
        self.assertEqual(entropy.shape, (batch_size, num_queries))
        self.assertEqual(probs.shape, (batch_size, num_queries, 4, n_bins))
        
        # Check range [0, 1]
        self.assertTrue(torch.all(coords >= 0))
        self.assertTrue(torch.all(coords <= 1))
        
        # Check entropy is non-negative
        self.assertTrue(torch.all(entropy >= 0))

    def test_sharp_distribution_entropy(self):
        # Construct a case with very sharp distribution (one hot)
        # Entropy should be close to 0
        hidden_dim = 256
        n_bins = 16
        head = DistributionHead(hidden_dim, n_bins)
        
        # Mock logits to be very high on one bin and low on others
        # We can simulate this by setting weights/bias or just testing the math function directly
        # Let's bypass the linear layer and test the math
        
        logits = torch.zeros(1, 1, 4, n_bins)
        logits[0,0,:,5] = 100.0 # High prob at bin 5
        
        probabilities = F.softmax(logits, dim=-1)
        # Entropy
        eps = 1e-8
        entropy = -torch.sum(probabilities * torch.log(probabilities + eps), dim=-1) # [1, 1, 4]
        
        # Should be close to 0
        self.assertTrue(torch.all(entropy < 0.01))
        
        # Expected value should be close to bins[5]
        expected_value = torch.sum(probabilities * head.bins, dim=-1)
        target_val = head.bins[5]
        
        self.assertTrue(torch.max(torch.abs(expected_value - target_val)) < 0.001)

    def test_flat_distribution_entropy(self):
        # Flat distribution (all zeros logits)
        hidden_dim = 256
        n_bins = 16
        head = DistributionHead(hidden_dim, n_bins)
        
        logits = torch.zeros(1, 1, 4, n_bins) # Uniform distribution
        probabilities = F.softmax(logits, dim=-1) # All 1/n_bins
        
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1)
        
        # Expected entropy for uniform is log(n_bins)
        expected_entropy = torch.log(torch.tensor(n_bins, dtype=torch.float))
        
        self.assertTrue(torch.max(torch.abs(entropy - expected_entropy)) < 0.001)

if __name__ == '__main__':
    unittest.main()
