import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import cat, stack, tensor
from numpy import sqrt


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class HaarDown(nn.Module):
    def __init__(self, inchan):
        super(HaarDown, self).__init__()
        self.haar_matrix = torch.tensor([[1., -1., -1., 1.],
                                         [1., -1., 1., -1.],
                                         [1., 1., -1., -1.]]).float().view(3, 2, 2) / 2.
        self.weights = torch.zeros([inchan*3, inchan, 2, 2])
        for i in xrange(inchan):
            self.weights[3*i:3*i+3, i] = self.haar_matrix

    def forward(self, x):
        if x.is_cuda:
            x = F.conv2d(x, self.weights.cuda(), stride=2)
        else:
            x = F.conv2d(x, self.weights, stride=2)
        return x


class HaarUp(nn.Module):
    def __init__(self, inchan):
        super(HaarUp, self).__init__()
        self.haar_matrix = torch.tensor([[1., -1., -1., 1.],
                                         [1., -1., 1., -1.],
                                         [1., 1., -1., -1.]]).float().view(3, 2, 2) / 2.
        self.weights = torch.zeros([inchan*3, inchan, 2, 2])
        for i in xrange(inchan):
            self.weights[3*i:3*i+3, i] = self.haar_matrix

    def forward(self, x):
        if x.is_cuda:
            x = F.conv_transpose2d(x, self.weights.cuda(), stride=2)
        else:
            x = F.conv_transpose2d(x, self.weights, stride=2)
        return x


class AvgUp(nn.Module):
    def __init__(self, upscale):
        super(AvgUp, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        s = x.size(1)
        repeat_idx = [1] * x.dim()
        repeat_idx[1] = self.upscale**2
        a = x.repeat(*(repeat_idx))
        order_index = torch.cat([s * torch.arange(self.upscale**2) + i for i in range(s)]).to(torch.long)
        x = torch.index_select(a, 1, order_index)
        return F.pixel_shuffle(x, upscale_factor=self.upscale)


class Haar(nn.Module):
    def __init__(self, inchan):
        super(Haar, self).__init__()
        self.haar = nn.Sequential(
            HaarDown(inchan),
            HaarUp(inchan)
        )


    def forward(self, x):
        x = self.haar(x)
        return x


class Down(nn.Module):
    def __init__(self, inchan, outchan):
        super(Down, self).__init__()
        self.d = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(inchan, outchan)
        )

    def forward(self, x):
        x = self.d(x)
        return x

class Up(nn.Module):
    def __init__(self, inchan, outchan, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(inchan//2, outchan//2, 2, stride=2)

        self.conv = DoubleConv(inchan, outchan)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        diffY2 = abs(x1.size()[2] - x3.size()[2])
        diffX2 = abs(x1.size()[3] - x3.size()[3])
        diffY1 = abs(x1.size()[2] - x2.size()[2])
        diffX1 = abs(x1.size()[3] - x2.size()[3])
        x2 = F.pad(x2, (diffX1 // 2, int(diffX1 / 2),
                        diffY1 // 2, int(diffY1 / 2)))
        x3 = F.pad(x3, (diffX2 // 2, int(diffX2 / 2),
                        diffY2 // 2, int(diffY2 / 2)))
        x = torch.cat([x3, x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, inchan, outchan):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(inchan * 2, outchan, 1)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class TightFrameUNet(nn.Module):
    def __init__(self, chan, residual=False):
        super(TightFrameUNet, self).__init__()
        self.inc = DoubleConv(chan, 64)
        self.outc = OutConv(64, chan)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.haar1 = Haar(64)
        self.haar2 = Haar(128)
        self.haar3 = Haar(256)
        self.haar4 = Haar(512)
        self.haar5 = Haar(512)
        self.up1 = Up(1536, 256)
        self.up2 = Up(1024, 128)
        self.up3 = Up(512, 64)
        self.up4 = Up(256, 64)
        self.residual = residual

    def forward(self, x):
        temp = x
        x1 = self.inc(x)            # B x 64  x H x W
        x2 = self.down1(x1)         # B x 128 x H x W
        x3 = self.down2(x2)         # B x 256 x H x W
        x4 = self.down3(x3)         # B x 512 x H x W
        x5 = self.down4(x4)         # B x 512 x H x W
        x1_haar = self.haar1(x1)
        x2_haar = self.haar2(x2)
        x3_haar = self.haar3(x3)
        x4_haar = self.haar4(x4)
        x5_haar = self.haar5(x5)

        x = self.up1(x5, x5_haar, x4)
        x = self.up2(x, x4_haar, x3 )
        x = self.up3(x, x3_haar, x2)
        x = self.up4(x, x2_haar, x1)
        if self.residual:
            x = self.outc(x1_haar, x) + temp
        else:
            x = self.outc(x1_haar, x)
        return x



