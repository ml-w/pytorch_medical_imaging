import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.conv = nn.Conv2d(inchan, inchan, 7, padding=3)
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