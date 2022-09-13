import torch
import torch.nn as nn

__all__ = ['ConfidenceBCELoss']

class ConfidenceBCELoss(nn.Module):
    def __init__(self, *args, conf_factor=0, **kwargs):
        r"""Assumes input from model have already passed through sigmoid"""
        super(ConfidenceBCELoss, self).__init__()
        self.base_loss = nn.BCELoss(*args, **kwargs, reduction='none')
        self.conf_factor = conf_factor
        self.conf_loss = nn.BCELoss(reduction='none')
        self.register_buffer('_epsilon', torch.DoubleTensor([1E-20]))

    def forward(self,
                input: torch.DoubleTensor,
                target: torch.DoubleTensor):
        # Typical classification loss for first dimension
        loss_classification = self.base_loss.forward(input[...,0].flatten(), target.flatten())
        loss_classification = torch.clamp(loss_classification, 0, 10).mean()


        # can the confidence score "predict right/wrong prediction"
        if input.shape[-1] >= 2:
            # If confidence is large and result is correctly predicted, adjust to lower loss, vice versa.
            # predict = torch.sigmoid(input[...,0].view_as(target)) > 0.5
            predict = input[...,0].view_as(target) > 0.5
            gt = target > 0
            correct_prediction = predict == gt

            loss_conf = self.conf_loss(input[..., 1].flatten(), correct_prediction.float().flatten())
            loss_conf = torch.clamp(loss_conf, 0, 10).mean()

            return (loss_classification + loss_conf * self.conf_factor) / (1 + self.conf_factor)
        else:
            return loss_classification

