import torch
import torch.nn as nn

__all__ = ['ConfidenceBCELoss']

class ConfidenceBCELoss(nn.Module):
    def __init__(self, *args, conf_factor=0.5, **kwargs):
        r"""Assumes input from model have not passed through sigmoid"""
        super(ConfidenceBCELoss, self).__init__()
        self.base_loss = nn.BCEWithLogitsLoss(*args, **kwargs)
        self.conf_factor = conf_factor

    def forward(self,
                input: torch.DoubleTensor,
                target: torch.DoubleTensor):
        # Typical classification loss for first dimension
        loss_classification = self.base_loss.forward(input[...,0].unsqueeze(-1), target)

        # If confidence is large and result is correctly predicted, adjust to lower loss, vice versa.
        predict = torch.sigmoid(input[..., 0]) > 0.5
        gt = target > 0
        conf_adjust = torch.ones_like(target) * -1
        conf_adjust[predict.view_as(gt) != gt] = 1
        loss_conf = conf_adjust.mul(torch.sigmoid(input[..., 1].unsqueeze(-1))).mean()

        return loss_classification + self.conf_factor * loss_conf
