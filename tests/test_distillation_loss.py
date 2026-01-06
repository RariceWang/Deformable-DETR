
import unittest
import torch
import torch.nn.functional as F

def loss_distill(outputs_student, outputs_teacher, temperature=1.0):
    """
    KL Divergence between student logits and teacher logits.
    """
    # logits: [B, N, 4*bins] -> view [B, N, 4, bins]
    logits_s = outputs_student
    logits_t = outputs_teacher
    
    B, N, C = logits_s.shape
    n_bins = C // 4
    
    logits_s = logits_s.view(B, N, 4, n_bins)
    logits_t = logits_t.view(B, N, 4, n_bins)
    
    # KL(P_t || P_s) or KL(P_s || P_t)?
    # Distillation usually minimizes KL(Teacher || Student) ? No, usually KL(Student || Teacher) -- wait.
    # P_t is fixed (target). We want P_s to approach P_t.
    # KL(P || Q) = sum P log (P/Q) = sum P log P - sum P log Q.
    # If we minimize w.r.t Q (Student), we minimize - sum P_t log Q_s.
    # This is Cross Entropy.
    
    # Standard KD: T^2 * KL(softmax(S/T), softmax(T/T))
    
    log_probs_s = F.log_softmax(logits_s / temperature, dim=-1)
    probs_t = F.softmax(logits_t / temperature, dim=-1)
    
    # KL Div Loss built-in expects input=log_probs, target=probs (for KLDivLoss with batchmean)
    # definition: l(x, y) = y * (log y - x)
    # x is log_prob (student), y is prob (teacher)
    
    loss = F.kl_div(log_probs_s, probs_t, reduction='batchmean') * (temperature ** 2)
    return loss

class TestDistillationLoss(unittest.TestCase):
    def test_distillation_value(self):
        B, N, bins = 1, 1, 4
        logits_s = torch.zeros(B, N, 4 * bins) # Uniform
        logits_t = torch.zeros(B, N, 4 * bins) # Uniform
        
        # If both identical, loss should be 0
        loss = loss_distill(logits_s, logits_t)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)
        
    def test_distillation_gradient(self):
        B, N, bins = 1, 1, 4
        logits_s = torch.randn(B, N, 4 * bins, requires_grad=True)
        logits_t = torch.randn(B, N, 4 * bins) # Fixed teacher
        
        loss = loss_distill(logits_s, logits_t)
        loss.backward()
        
        self.assertIsNotNone(logits_s.grad)

if __name__ == '__main__':
    unittest.main()
