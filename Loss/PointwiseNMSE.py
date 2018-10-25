import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: something wrong with this loss function in 0.4.1, something to do with size average
class PointWiseNMSELoss(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(PointWiseNMSELoss, self).__init__()
        self._size_average=size_average
        self._reduce = reduce

    def forward(self, input, output, target):
        mseloss = F.mse_loss(output, target, reduce=False)
        factor = F.mse_loss(input, target, reduce=False)
        d = mseloss / factor
        if not self._reduce:
            return d
        else:
            return torch.mean(d) if self._size_average else torch.sum(d)