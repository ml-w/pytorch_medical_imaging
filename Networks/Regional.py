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

class StrideConv(nn.Module):
    def __init__(self, inchan, outchan, kernsize=5, stride=2):
        super(StrideConv, self).__init__()
        self.dconv = nn.Conv3d(inchan, outchan, kernsize, stride=tuple(stride), padding=(kernsize - 1) /2)
        self.bn = nn.BatchNorm3d(outchan)

    def forward(self, x):
        return self.bn(self.dconv(x))


class Shallow(nn.Module):
    def __init__(self):
        super(Shallow, self).__init__()
        self.conv1 = StandardConv(1, 16, 5)
        self.conv2 = StandardConv(16, 32, 5)
        self.conv3 = StandardConv(32, 64, 5)
        self.conv4 = StandardDeConv(64, 32, 5)
        self.conv5 = StandardDeConv(32, 16, 5)
        self.conv6 = StandardDeConv(16, 1, 5)
        self.initbn = nn.BatchNorm3d(1)

    def forward(self, x):
        x = self.initbn(x)
        x = self.conv1(F.max_pool3d(x, kernel_size=[1, 2, 2]))
        x = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(x)))))
        x = F.upsample(x, scale_factor=[1, 2, 2], mode='trilinear')
        return x


class MultiScale(nn.Module):
    def __init__(self):
        super(MultiScale, self).__init__()
        self.conv1 = StandardConv(1, 16, 5, padding=True)
        self.poolConv1 = StrideConv(16, 32, 5, (1, 2, 2))
        self.conv2 = StandardConv(32, 64, 5, padding=True)
        self.conv3 = StandardConv(64, 128, 5, padding=True)
        self.poolConv2 = StrideConv(128, 256, 3, (2, 2, 2))
        self.conv4 = StandardConv(256, 512, 3, padding=True)
        self.conv5 = StandardConv(512, 256, 3, padding=True)
        self.conv6 = StandardConv(256, 128, 3, padding=True)

        self.conv7 = StandardConv(128, 16, 3, padding=True)
        self.conv8 = StandardConv(16, 1, 3, padding=True)

        self.initbn = nn.BatchNorm3d(1)

    def forward(self, x):
        k = self.initbn(x)
        k = self.conv1(F.max_pool3d(k, kernel_size=[1, 2, 2]))
        k1 = self.conv3(self.conv2(self.poolConv1(k)))
        k2 = self.conv6(self.conv5(self.conv4(self.poolConv2(k1))))
        k1 = F.upsample(k2, scale_factor=2, mode='nearest') + k1
        k = F.upsample(self.conv7(k1), scale_factor=[1, 2, 2], mode='trilinear') + k
        x = F.upsample(self.conv8(k), scale_factor=[1, 2, 2], mode='trilinear')
        # x = k - x
        return x




class ShallowMasked(nn.Module):
    def __init__(self, indim):
        super(ShallowMasked, self).__init__()

        assert len(indim) == 2
        d = [1, 1, 1]
        d.extend(indim)
        self.weightmask = nn.Parameter(torch.ones(d))
        self.conv1 = StandardConv(1, 16, 5)
        self.conv2 = StandardConv(16, 32, 5)
        self.conv3 = StandardConv(32, 64, 5)
        self.conv4 = StandardDeConv(64, 64, 5)
        self.conv5 = StandardDeConv(64, 32, 5)
        self.conv6 = StandardDeConv(32, 1, 5)
        self.initbn = nn.BatchNorm3d(1)

    def forward(self, x):
	print torch.sum(self.weightmask).data[0]
        x = x * self.weightmask.expand_as(x)
        x = self.initbn(x)
        x = self.conv1(F.max_pool3d(x, kernel_size=[1, 4, 4]))
        x = self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(x)))))
        x = F.upsample(x, scale_factor=[1, 4, 4], mode='trilinear')
        return x

