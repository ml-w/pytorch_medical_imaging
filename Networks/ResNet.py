import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class Kernel(nn.Module):
    def __init__(self, inchan, outchan, kernsize=5):
        super(Kernel, self).__init__()
        self.conv = nn.Conv2d(inchan, outchan, kernsize, padding=(kernsize - 1)/2 )
        self.bn = nn.BatchNorm2d(outchan)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x

class ResKernel(nn.Module):
    def __init__(self, inchan):
        super(ResKernel, self).__init__()
        self.k1 = Kernel(inchan, inchan)
        self.conv = nn.Conv2d(inchan, inchan, [3, 7], padding=(1,3))
        self.bn = nn.BatchNorm2d(inchan)

    def forward(self, x):
        c = self.k1(x)
        c = self.bn(self.conv(c))
        c = c + x
        c = F.relu(c)
        return c

class ResNet(nn.Module):
    def __init__(self, inchan, outchan, depth):
        super(ResNet, self).__init__()
        self.depth = depth
        self.features = 64
        self.upscale = 8

        self.inBn = nn.BatchNorm2d(1)
        self.initkern = Kernel(inchan, self.features)
        self.kerns1 = nn.Sequential(*[ResKernel(self.features) for i in xrange((depth - 1)/2)])
        # self.kerns2 = nn.Sequential(*[ResKernel(self.features/(self.upscale**2))
        #                               for i in xrange((depth - 1)/4)])
        # self.outkern = Kernel(self.features/(self.upscale**2), outchan)
        self.outConv = nn.Conv2d(1, 1, self.upscale, stride=self.upscale)
        self.bnout = nn.BatchNorm2d(1)

    def forward(self, x):
        c = self.inBn(x)
        c = self.initkern(c)
        c = self.kerns1.forward(c)
        c = F.pixel_shuffle(c, self.upscale)
        c = self.outConv(c)
        c = self.bnout(c)
        # c = self.kerns2.forward(c)
        # c = self.outkern(c)
        # c = F.avg_pool2d(c, self.upscale)
        c = c + x
        return c


class ResNetB(nn.Module):
    def __init__(self, inchan, outchan, depth):
        super(ResNetB, self).__init__()
        self.upscale = int(math.sqrt(inchan))
        self.depth = depth
        self.kerns1 = nn.Sequential(*[ResKernel(inchan) for i in xrange((depth - 1)/2)])
        self.outConv = nn.Conv2d(1, outchan, self.upscale, stride=self.upscale)
        self.bnout = nn.BatchNorm2d(outchan)

    def forward(self, x):
        c = self.kerns1.forward(x)
        c = F.pixel_shuffle(c, self.upscale)
        c = self.outConv(c)
        c = self.bnout(c)
        return c


class ADResNet(nn.Module):
    def __init__(self, inchan, outchan, depth):
        super(ADResNet, self).__init__()
        self.features = 49

        self.res1 = ResNetB(self.features, outchan, depth)
        self.res2 = ResNetB(self.features, outchan, depth)
        self.initbn = nn.BatchNorm2d(inchan)
        self.initkern = Kernel(inchan, self.features)

    def forward(self, x, ratio):
        t = self.initkern(self.initbn(x))
        t1 = self.res1.forward(t) + x
        t2 = self.res2.forward(t) + x
        t1 = t1.transpose(0, -1)
        t2 = t2.transpose(0, -1)
        t = ratio[:,0].expand_as(t1) * t1  + ratio[:,1].expand_as(t2) * t2
        t = t.transpose(0, -1)
        return t