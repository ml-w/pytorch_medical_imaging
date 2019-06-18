from Layers import Conv3d
import torch
import torch.nn as nn
import torch.nn.functional as F

class LLinDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LLinDoubleConv, self).__init__()
        nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch)
        )

    def forward(self, x):
        return self.conv(x)

class LLinResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LLinResidualBlock, self).__init__()
        self.conv = LLinDoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x) + x


class LLinNet(nn.Module):
    def __init__(self, in_ch):
        super(LLinNet, self).__init__()
        self.inconv = nn.Conv3d(in_ch, 64)


        self.res1 = LLinResidualBlock(64, 64)
        self.res2 = LLinResidualBlock(64, 64)
        self.res3 = LLinResidualBlock(64, 64)
        self.res4 = LLinResidualBlock(64, 64)
        self.res5 = LLinResidualBlock(64, 64)
        self.res6 = LLinResidualBlock(64, 64)

        self.out1 = LLinDoubleConv(64, 64)
        self.out2 = LLinDoubleConv(64, 64)
        self.out3 = LLinDoubleConv(64, 64)

    def forward(self, x):

        x0 = F.max_pool3d(self.inconv(x))

        x1 = F.max_pool3d(self.res2(self.res1(x0)))
        x2 = F.max_pool3d(self.res4(self.res3(x1)))
        x3 = F.max_pool3d(self.res6(self.res5(x2)))

        u1 = self.out1(torch.cat([x1, F.upsample_nearest(x3, 2)], dim=2))
        u2 = self.out2(torch.cat([x0, F.upsample_nearest(u1, 2)], dim=2))
        out = self.out3(F.upsample_nearest((u2, 2)))
        out = F.softmax(out)
        return out


