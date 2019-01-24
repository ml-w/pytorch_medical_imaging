'''
From https://github.com/milesial/Pytorch-UNet
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from Layers import ReflectiveDoubleConv as double_conv

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, max_pool=False):
        super(down, self).__init__()

        if max_pool:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch)
            )
        else:
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
            self.up = lambda input: nn.functional.interpolate(input,
                                                              scale_factor=2,
                                                              mode='bilinear',
                                                              align_corners=False)
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

class UNetLocTexAware(UNet):
    def __init__(self, *args, **kwargs):
        super(UNetLocTexAware, self).__init__(*args, **kwargs)

        self.inc = nn.Sequential(
            nn.Conv2d(args[0], 128, groups=2, kernel_size=3, padding=1),
            nn.Conv2d(128, 64, kernel_size=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(4, 256),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
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

        if self.residual:
            x = self.outc(x) + temp
        else:
            x = self.outc(x)

        # expand pos
        pos = self.fc(pos)
        pos = pos.expand_as(x.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
        x = x * pos

        return x


class UNetLocTexAware2(UNet):
    def __init__(self, *args, **kwargs):
        super(UNetLocTexAware2, self).__init__(*args, **kwargs)

        self.fc = nn.Sequential(
            nn.Linear(104, 300),
            nn.LayerNorm(300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 600),
            nn.LayerNorm(600),
            nn.ReLU(inplace=True)
        )
        self.fc5 = nn.Linear(600, 512)
        self.fc4 = nn.Linear(600, 512)
        self.fc3 = nn.Linear(600, 256)
        self.fc2 = nn.Linear(600, 128)

        self.outc = nn.Conv2d(64, 2, 1)
        self.dropout1 = nn.Dropout2d(0.2, inplace=False)
        self.dropout2 = nn.Dropout2d(0.3, inplace=False)


    def forward(self, x, pos):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # expand pos
        pos = self.fc(pos)

        X = []
        for _x, _fc in zip([x2, x3, x4, x5], [self.fc2, self.fc3, self.fc4, self.fc5]):
            _pos = _fc(pos).expand_as(_x.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
            _x = _x * F.relu(_pos, True)
            X.append(_x)
        x2, x3, x4, x5 = X

        x = self.up1(self.dropout2(x5), self.dropout1(x4))
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x)
        return x

class UNetLocTexAwareDeeper(UNet):
    def __init__(self, *args, **kwargs):
        super(UNetLocTexAwareDeeper, self).__init__(*args, **kwargs)

        self.fc = nn.Sequential(
            nn.Linear(104, 300),
            nn.LayerNorm(300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 600),
            nn.LayerNorm(600),
            nn.ReLU(inplace=True)
        )

        self.down4 = down(512, 1024, max_pool=True)
        self.down5 = down(1024, 1024, max_pool=True)

        self.up0 = up(2048, 512, True)

        self.fc6 = nn.Linear(600, 1024)
        self.fc5 = nn.Linear(600, 1024)
        self.fc4 = nn.Linear(600, 512)
        self.fc3 = nn.Linear(600, 256)
        self.fc2 = nn.Linear(600, 128)

        self.outc = nn.Conv2d(64, 2, 1)
        self.dropout1 = nn.Dropout2d(0.2, inplace=False)
        self.dropout2 = nn.Dropout2d(0.3, inplace=False)


    def forward(self, x, pos):
        x1 = self.inc(x)        # 128
        x2 = self.down1(x1)     # 64
        x3 = self.down2(x2)     # 32
        x4 = self.down3(x3)     # 16
        x5 = self.down4(x4)     # 8
        x6 = self.down5(x5)     # 4
        # expand pos
        pos = self.fc(pos)

        X = []
        for _x, _fc in zip([x2, x3, x4, x5, x6], [self.fc2, self.fc3, self.fc4, self.fc5, self.fc6]):
            _pos = _fc(pos).expand_as(_x.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
            _x = _x * F.relu(_pos, True)
            X.append(_x)
        x2, x3, x4, x5, x6 = X

        x = self.up0(self.dropout2(x6), self.dropout1(x5))
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
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
