from Layers import Conv3d
import torch
import torch.nn as nn
import torch.nn.functional as F

class LLinDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernsize):
        super(LLinDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, kernsize, padding=kernsize - 2),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernsize, padding=kernsize - 2)
        )

    def forward(self, x):
        return self.conv(x)

class LLinResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LLinResidualBlock, self).__init__()
        self.conv = LLinDoubleConv(in_ch, out_ch, 3)

    def forward(self, x):
        return self.conv(x) + x


class LLinNet(nn.Module):
    def __init__(self, in_ch, num_of_class):
        super(LLinNet, self).__init__()
        self.inconv = nn.Conv3d(in_ch, 64, 3, padding=1)


        self.res1 = LLinResidualBlock(64, 64)
        self.res2 = LLinResidualBlock(64, 64)
        self.res3 = LLinResidualBlock(64, 64)
        self.res4 = LLinResidualBlock(64, 64)
        self.res5 = LLinResidualBlock(64, 64)
        self.res6 = LLinResidualBlock(64, 64)

        self.out1 = LLinDoubleConv(128, 64, 3)
        self.out2 = LLinDoubleConv(128, 64, 3)
        self.out3 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, num_of_class, 3, padding=1)
        )

    def forward(self, x):
        x0 = F.max_pool3d(self.inconv(x), 2)            # 8 x 24 x 24
        x1 = F.max_pool3d(self.res2(self.res1(x0)), 2)  # 4 x 12 x 12
        x2 = F.max_pool3d(self.res4(self.res3(x1)), 2)  # 2 x 6 x 6
        x3 = self.res6(self.res5(x2))                   # 2 x 6 x 6

        u1 = self.out1(torch.cat([x1, F.interpolate(x3, scale_factor=2)], dim=1))   # 128 x 4 x 12 x 12
        u2 = self.out2(torch.cat([x0, F.interpolate(u1, scale_factor=2)], dim=1))   # 128 x 8 x 24 x 24
        out = self.out3(F.interpolate(u2, scale_factor=2))  # 2 x 16 x 48 x 48
        # out = F.softmax(out)
        return out


