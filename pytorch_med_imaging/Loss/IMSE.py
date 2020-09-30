import torch
import torch.nn as nn
import torch.nn.functional as F

class IMSELoss(nn.Module):
    def __init__(self, size_average=False, reduce=True):
        super(IMSELoss, self).__init__()
        self._size_average=size_average
        self._reduce = reduce

    def forward(self, input, raw, target):
        mseloss = F.mse_loss(target, input,
                             size_average = self._size_average,
                             reduce = self._reduce)
        factor = F.mse_loss(raw, target,
                            size_average=self._size_average,
                            reduce = self._reduce)
        return mseloss / factor


