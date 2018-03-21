import torch
import torch.nn as nn
import torch.nn.functional as F


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

class Shallow(nn.Module):
    def __init__(self):
        super(Shallow, self).__init__()
        self.conv1 = StandardConv(1, 32, 5, True)
        self.conv2 = StandardConv(32, 64, 5, True)
        self.conv3 = StandardConv(64, 128, 5, True)
        self.conv4 = StandardConv(128, 64, 3, True)
        self.conv5 = StandardConv(64, 1, 3, True)
        self.initbn = nn.BatchNorm3d(1)

    def forward(self, x):
        x = self.initbn(x)
        x = self.conv1(F.max_pool3d(x, kernel_size=[1, 2, 2]))
        x = self.conv2(F.max_pool3d(x, 2))
        x = self.conv5(self.conv4(self.conv3(x)))
        x = F.upsample(x, scale_factor=2, mode='trilinear')
        x = F.upsample(x, scale_factor=[1, 2, 2], mode='trilinear')
        return x
