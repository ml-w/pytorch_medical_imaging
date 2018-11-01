import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: something wrong with this loss function in 0.4.1, something to do with size average
class PointwiseNMSELoss(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(PointwiseNMSELoss, self).__init__()
        self._size_average=size_average
        self._reduce = reduce

    def forward(self, input, output, target):
        mseloss = F.l1_loss(output, target, reduce=False)
        factor = F.l1_loss(input, target, reduce=False)
        factor = torch.clamp(factor, 1, 1E30)
        d = mseloss / factor
        if not self._reduce:
            return d
        else:
            return torch.mean(d) if self._size_average else torch.sum(d)