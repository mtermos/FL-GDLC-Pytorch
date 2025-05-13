import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi‐class classification.

    Args:
        alpha (float): balance factor, multiplies the focal term.
        gamma (float): focusing parameter; higher ⇒ more focus on hard examples.
        reduction (str): 'none' | 'mean' | 'sum'
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: shape (batch_size, num_classes)
        targets: shape (batch_size,) with class indices
        """
        # per‐sample cross‐entropy without reduction
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        # pt = probability assigned to the true class
        pt = torch.exp(-ce_loss)
        # focal scaling
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
