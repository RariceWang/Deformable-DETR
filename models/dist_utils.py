import torch
import torch.nn.functional as F

class DistributionHead(torch.nn.Module):
    def __init__(self, reg_max=32, reg_scale=1.0):
        super().__init__()
        self.reg_max = reg_max
        # Define bins: 0 to reg_max
        # We will map bin index i to value: i / reg_max
        # So value range is [0, 1]
        self.register_buffer('project', torch.linspace(0, 1, reg_max + 1))

    def forward(self, x):
        """
        x: [..., 4 * (reg_max + 1)]
        Returns:
            coords: [..., 4] (l, t, r, b) or (cx, cy, w, h) depending on usage
            entropy: [..., 4] entropy of each coordinate distribution
        """
        shape = x.shape
        # reshape to [..., 4, bins]
        x_reshaped = x.reshape(shape[:-1] + (4, self.reg_max + 1))
        
        # Softmax to get probabilities
        prob = F.softmax(x_reshaped, dim=-1) # [..., 4, bins]
        
        # Calculate Expected Value (Integration)
        # sum(p * val)
        # project is [bins]
        val = torch.sum(prob * self.project, dim=-1) # [..., 4]
        
        # Calculate Entropy
        # H = -sum(p * log(p))
        # Add epsilon to prevent log(0)
        log_prob = torch.log(prob + 1e-8)
        entropy = -torch.sum(prob * log_prob, dim=-1) # [..., 4]
        
        return val, entropy, prob

def dist_l1_loss(pred_boxes, target_boxes):
    return F.l1_loss(pred_boxes, target_boxes, reduction='none')
