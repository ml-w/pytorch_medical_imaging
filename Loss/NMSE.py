import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class NMSELoss(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(NMSELoss, self).__init__()
        self._size_average=size_average
        self._reduce = reduce

    def forward(self, input, target):
        mseloss = F.mse_loss(target, input,
                             size_average = self._size_average,
                             reduce = self._reduce)
        zeros = Variable(torch.zeros(input.size()), requires_grad=False)
        if self.cuda():
            zeros = zeros.cuda()
        factor = F.mse_loss(target, zeros,
                            size_average=self._size_average,
                            reduce = self._reduce)
        return mseloss / factor

