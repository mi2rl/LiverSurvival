import torch
import torch.nn as nn


class surv_Loss(nn.Module):
    def __init__(self, device):
        super(surv_Loss, self).__init__()
        self.device = device

    def forward(self, surv_s, surv_f, surv_pred):
        surv_s = surv_s.float()
        surv_f = surv_f.float()
        surv_pred = surv_pred.float()

        first = 1+(surv_s*(surv_pred-1))
        first = torch.clamp(first, min=1e-7)
        first = torch.log(first)

        second = 1-surv_f*surv_pred
        second = torch.clamp(second, min=1e-7)
        second = torch.log(second)

        result = torch.add(first, second)
        result = torch.sum(result, dim=-1)
        result = torch.mean(result)
        return -result

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross-entropy loss."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        confidence = 1. - self.smoothing
        log_probs = F.log_softmax(logits, dim=-1)
        true_probs = torch.full_like(log_probs, self.smoothing / (log_probs.size(1) - 1))
        true_probs.scatter_(1, targets.unsqueeze(1), confidence)
        return (-true_probs * log_probs).sum(dim=1).mean()