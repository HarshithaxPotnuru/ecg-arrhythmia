import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            reduction='none',
            pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce)
        loss = ((1 - pt) ** self.gamma) * bce
        return loss.mean()
