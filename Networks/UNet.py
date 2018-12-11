'''
From https://github.com/milesial/Pytorch-UNet
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from Layers import DoubleConv as double_conv

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.AvgPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, outchan=None, residual=True):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, True)
        self.up2 = up(512, 128, True)
        self.up3 = up(256, 64, True)
        self.up4 = up(128, 64, True)
        self.outc = outconv(64, n_channels) if outchan is None else outconv(64, outchan)
        self.residual= residual

    def forward(self, x):
        temp = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.residual:
            x = self.outc(x) + temp
        else:
            x = self.outc(x)
        return x

class UNetPosAware(UNet):
    def __init__(self, *args, **kwargs):
        super(UNetPosAware, self).__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(1, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64)
        )
        self.outc = nn.Conv2d(64, 2, 1)


    def forward(self, x, pos):
        temp = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # expand pos
        pos = self.fc(pos.view(len(pos), 1))
        pos = pos.expand_as(x.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
        x = x * pos
        if self.residual:
            x = self.outc(x) + temp
        else:
            x = self.outc(x)
        return x

class UNetSubbands(nn.Module):
    def __init__(self, inchan):
        super(UNetSubbands, self).__init__()
        self.chan = inchan
        if inchan > 1:
            self.net1 = UNet(inchan//2)
            self.net2 = UNet(inchan//2)
        else:
            self.net = UNet(1)

    def forward(self, x):
        if self.chan > 1:
            x1 = self.net1(x.narrow(1, 0, self.chan//2))
            x2 = self.net2(x.narrow(1, self.chan//2, self.chan//2))
            return torch.cat([x1, x2], dim=1)
        else:

            return self.net.forward(x)
