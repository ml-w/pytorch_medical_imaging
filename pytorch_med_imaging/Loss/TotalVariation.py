import torch
import torch.nn as nn
import torch.nn.functional as F

class TV(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(TV, self).__init__()
        self._reduce = reduce
        self._size_average = size_average
        self._vkern = torch.tensor([[1], [-1], [0]])
        self._hkern = torch.tensor([[1, -1, 0]])


    def forward(self, input):
        b, c, h, w = input.size()
        vweights = torch.zeros([c, c, 3, 1])
        hweights = torch.zeros([c, c, 1, 3])
        for i in range(c):
            vweights[i, i] = self._vkern
            hweights[i, i] = self._hkern

        if input.is_cuda:
            vdiff = F.conv2d(input, vweights.cuda(), padding=(1, 0))
            hdiff = F.conv2d(input, hweights.cuda(), padding=(0, 1))
        else:
            vdiff = F.conv2d(input, vweights, padding=(1, 0))
            hdiff = F.conv2d(input, hweights, padding=(0, 1))

        tv = torch.abs(vdiff) + torch.abs(hdiff)
        if self._reduce:
            tv = torch.sum(tv, 1)

        del vweights, hweights
        return tv


class TVLoss(nn.Module):
    def __init__(self, size_average=True, reduce=True):
        super(TVLoss, self).__init__()
        self._size_average = size_average
        self._reduce = reduce
        self.tv0 = TV(False)
        self.tv1 = TV(False)

    def forward(self, input, target):
        intv = self.tv0(input)
        tartv = self.tv1(input)

        out = torch.abs(intv - tartv)

        if self._reduce:
            return torch.mean(out) if self._size_average else torch.sum(out)
        else:
            return out
