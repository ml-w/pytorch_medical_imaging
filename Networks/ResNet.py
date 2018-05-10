import torch
import torch.nn as nn
import torch.nn.functional as F
import visdom
# testing
import numpy as np

vis = visdom.Visdom(port=80)


class ResidualDownTransition(nn.Module):
    def __init__(self, feqtureSize, kernsize):
        super(ResidualDownTransition, self).__init__()

        self.feqtureSize = feqtureSize

        pad = np.int(kernsize/2.)
        self.conv1 = nn.Conv2d(feqtureSize, feqtureSize, kernel_size=kernsize,padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(feqtureSize)
        self.conv2 = nn.Conv2d(feqtureSize, feqtureSize, kernel_size=kernsize,padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(feqtureSize)

    def forward(self, x):

        down = F.relu(self.bn1(self.conv1(x)))
        down = F.relu(self.bn2(self.conv2(down)) + x)
        return down

class WeightedSum(nn.Module):
    def __init__(self, channels, positive=True):
        super(WeightedSum, self).__init__()

        self.channels = channels
        self.positive = positive
        self.params = nn.ParameterList()
        self.params.extend([nn.Parameter(torch.ones(1)) for i in xrange(channels)])


    def forward(self, x):
        assert x.data.size()[1] == self.channels, "Wrong number of channels!"

        tempx = x[:, 0] * self.params[0].expand_as(x[:,0])
        for i in xrange(1, self.channels):
            if self.positive:
                tempx = tempx + x[:,i] * torch.abs(self.params[i]).expand_as(x[:,i])
            else:
                tempx = tempx + x[:,i] * self.params[i].expand_as(x[:,i])
        x = tempx / self.channels
        return x

class DownTransition(nn.Module):
    def __init__(self, inchan, outchan, kernsize, padding=True):
        super(DownTransition, self).__init__()

        self.inchan = inchan
        self.outchan = outchan
        self.padding = padding
        self.kernsize = kernsize

        if padding:
            pad = np.int(kernsize/2.)
        else:
            pad = 0

        self.conv1 = nn.Conv2d(inchan, outchan, kernel_size=kernsize,padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(outchan)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return x

class UpTransition(nn.Module):
    def __init__(self, upscaleFactor):
        super(UpTransition, self).__init__()

        self.ps = nn.PixelShuffle(upscaleFactor)
        self.pool = nn.AvgPool2d(upscaleFactor)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.pool(self.ps(x))
        x = self.bn(x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.kernsize = 7
        self.chansize = 32
        self.num_of_layers = 9

        self.convsModules = nn.ModuleList()
        self.psModules = nn.ModuleList()
        self.fcModules = nn.ModuleList()
        self.bnModules = nn.ModuleList()
        self.poolingLayers = nn.ModuleList()
        self.linearModules = nn.ModuleList()
        self.miscParams = nn.ParameterList()
        self.initBn = nn.BatchNorm2d(1)

        self.d_32 = DownTransition(1, self.chansize, self.kernsize)
        self.DTrans = [ResidualDownTransition(self.chansize, self.kernsize)
                       for i in xrange(self.num_of_layers - 1)]
        self.d_36 = DownTransition(self.chansize, 36, self.kernsize)
        self.u   = UpTransition(np.int(np.sqrt(36)))

        self.convsModules.extend([d for d in self.DTrans])
        self.psModules.append(self.u)

        # self.CircularMask = None

        self.current_step = 0
        self.current_epoch = 0
        self.loss_list = []

    def forward(self, x):
        """
        x2 should have better resolution
        :param x1:
        :param x2:
        :return:
        """
        assert x.is_cuda, "Inputs are not in GPU!"

        orix = x * 1
        x = self.initBn(x)

        x = self.d_32.forward(x)
        for i in xrange(self.num_of_layers - 1):
            x = self.DTrans[i].forward(x)
        x = self.d_36.forward(x)
        x = self.u.forward(x)
        x = x + orix

        # s = x.data.size()
        # if self.CircularMask is None:
        #     self.CircularMask = []
        #     for i in xrange(s[-2]):
        #         for j in xrange(s[-1]):
        #             if (i - s[-2]/2.) ** 2 + (j - s[-1] / 2.) ** 2 > ((s[-2]/2.)**2.) + 1:
        #                 self.CircularMask.append([i,j])
        #
        # for c in self.CircularMask:
        #     x[:,c[0],c[1]] = 0

        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


