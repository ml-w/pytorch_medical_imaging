import torch
import torch.nn as nn

from .Layers import DenseBlock, DenseConv

__all__ = ['DenseUNet2D']

class Transition(nn.Module):
    def __init__(self, n_channels):
        super(Transition, self).__init__()
        self.conv = nn.Sequential(
            DenseConv(n_channels, n_channels, 1),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, n_channels, k, num_layers):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            Transition(n_channels),
            DenseBlock(n_channels, k, num_layers)
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, inchan, outchan, bothup=False):
        super(Up, self).__init__()

        self.up = bothup
        self.upsample =nn.Upsample(scale_factor=2, align_corners=True, mode='bilinear')
        self.conv = DenseConv(inchan, outchan, 3)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        if self.up:
            x2 = self.upsample(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class DenseUNet2D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(DenseUNet2D, self).__init__()

        k = 32
        init_conv_out = 96
        self.inconv = DenseConv(n_channels, init_conv_out, 7, stride=2, pad=True)
        self.pooling = nn.MaxPool2d(3, stride=2, padding=1)
        self.down1 = DenseBlock(init_conv_out, k, 6)
        self.down2 = Down(init_conv_out + 6 * k, k, 12)
        self.down3 = Down(init_conv_out + 18* k, k, 36)
        self.down4 = Down(init_conv_out + 54* k, k, 24)

        self.up1 = Up(init_conv_out * 2 + 132 * k, 768)
        self.up2 = Up(768 + init_conv_out + 18* k, 384)
        self.up3 = Up(384 + init_conv_out + 6 * k, 96)
        self.up4 = Up(96  + init_conv_out, 96)
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, align_corners=True, mode='bilinear'),
            DenseConv(96, 64)
        )
        self.lastup = DenseConv(64, n_classes, 1)

    def forward(self, x):
        d1 = self.inconv(x)
        # print d1.shape
        d2 = self.down1(self.pooling(d1))
        # print d2.shape
        d3 = self.down2(d2)
        # print d3.shape
        d4 = self.down3(d3)
        # print d4.shape
        d5 = self.down4(d4)
        # print d5.shape
        x = self.up1(d5, d4)
        # print x.shape
        x = self.up2(x, d3)
        # print x.shape
        x = self.up3(x, d2)
        # print x.shape
        x = self.up4(x, d1)
        # print x.shape
        x = self.up5(x)
        # print x.shape
        x = self.lastup(x)
        # print x.shape
        return x
