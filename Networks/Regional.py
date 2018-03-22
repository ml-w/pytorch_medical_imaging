import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class StandardConv(nn.Module):
    def __init__(self, inchan, outchan, kernsize=5, padding=False):
        super(StandardConv, self).__init__()
        if padding:
            self.conv = nn.Conv3d(inchan, outchan, kernsize, padding=(kernsize - 1) /2)
        else:
            self.conv = nn.Conv3d(inchan, outchan, kernsize)
        self.bn = nn.BatchNorm3d(outchan)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class StandardDeConv(nn.Module):
    def __init__(self, inchan, outchan, kernsize=5):
        super(StandardDeConv, self).__init__()
        self.dconv = nn.ConvTranspose3d(inchan, outchan, kernsize)
        self.bn = nn.BatchNorm3d(outchan)

    def forward(self, x):
        return self.bn(self.dconv(x))


class Shallow(nn.Module):
    def __init__(self):
        super(Shallow, self).__init__()
        self.conv1 = StandardConv(1, 32, 5)
        self.conv2 = StandardConv(32, 64, 5)
        self.conv3 = StandardConv(64, 128, 5)
        self.conv4 = StandardDeConv(128, 64, 5)
        self.conv5 = StandardDeConv(64, 32, 5)
        self.conv6 = StandardDeConv(32, 1, 5)
        self.initbn = nn.BatchNorm3d(1)

    def forward(self, x):
        x = self.initbn(x)
        x = self.conv1(F.max_pool3d(x, kernel_size=[1, 4, 4]))
        x = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(x)))))
        x = F.upsample(x, scale_factor=[1, 4, 4], mode='trilinear')
        return x

class ShallowMasked(nn.Module):
    def __init__(self, indim):
        super(ShallowMasked, self).__init__()

        assert len(indim) == 2
        d = [1, 1, 1].extend(indim)
        self.weightmask = Variable(torch.ones(d), requires_grad=True)
        self.conv1 = StandardConv(1, 16, 5)
        self.conv2 = StandardConv(16, 32, 5)
        self.conv3 = StandardConv(32, 64, 5)
        self.conv4 = StandardDeConv(64, 64, 5)
        self.conv5 = StandardDeConv(64, 32, 5)
        self.conv6 = StandardDeConv(32, 1, 5)
        self.initbn = nn.BatchNorm3d(1)

    def forward(self, x):
        x = x * self.weightmask.expand_as(x)
        x = self.initbn(x)
        x = self.conv1(F.max_pool3d(x, kernel_size=[1, 4, 4]))
        x = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(x)))))
        x = F.upsample(x, scale_factor=[1, 4, 4], mode='trilinear')
        return x