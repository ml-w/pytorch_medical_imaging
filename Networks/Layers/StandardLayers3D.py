import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kern_size=3, stride=1, padding=1):
        super(Conv3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kern_size, stride, padding=padding),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv3d, self).__init__()
        self.conv = nn.Sequential(
            Conv3d(in_ch, out_ch),
            Conv3d(out_ch, out_ch)
        )


    def forward(self, x):
        return self.conv(x)


class ResidualBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResidualBlock3d, self).__init__()

        self.conv = nn.Sequential(
            DoubleConv3d(in_ch, out_ch)
        )
        self.bn = nn.BatchNorm3d(in_ch)