import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import cat, stack, tensor
from numpy import sqrt

from .Layers import DoubleConv2d, CircularDoubleConv


class HaarDown(nn.Module):
    def __init__(self, inchan):
        super(HaarDown, self).__init__()
        self.haar_matrix = torch.tensor([[1., -1., -1., 1.],
                                         [1., -1., 1., -1.],
                                         [1., 1., -1., -1.]]).float().view(3, 2, 2) / 2.
        self.weights = torch.zeros([inchan*3, inchan, 2, 2])
        for i in range(inchan):
            self.weights[3*i:3*i+3, i] = self.haar_matrix

    def forward(self, x):
        if x.is_cuda:
            x = F.conv2d(x, self.weights.cuda(), stride=2)
        else:
            x = F.conv2d(x, self.weights, stride=2)
        return x


class HaarUp(nn.Module):
    def __init__(self, inchan, reduce=True):
        super(HaarUp, self).__init__()
        self.haar_matrix = torch.tensor([[1., -1., -1., 1.],
                                         [1., -1., 1., -1.],
                                         [1., 1., -1., -1.]]).float().view(3, 2, 2) / 2.

        if reduce:
            self.weights = torch.zeros([inchan*3, inchan, 2, 2])
            for i in range(inchan):
                self.weights[3*i:3*i+3, i] = self.haar_matrix
        else:
            self.weights = torch.zeros([inchan*3, inchan*3, 2, 2])
            for i in range(inchan):
                self.weights[i, 3*i:3*i+3] = self.haar_matrix

    def forward(self, x):
        if x.is_cuda:
            x = F.conv_transpose2d(x, self.weights.cuda(), stride=2)
        else:
            x = F.conv_transpose2d(x, self.weights, stride=2)
        return x


class AvgUp(nn.Module):
    def  __init__(self, upscale):
        super(AvgUp, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        s = x.size(1)
        repeat_idx = [1] * x.dim()
        repeat_idx[1] = self.upscale**2
        a = x.repeat(*(repeat_idx))
        order_index = torch.cat([s * torch.arange(self.upscale**2) + i for i in range(s)]).to(torch.long)
        if x.is_cuda:
            x = torch.index_select(a, 1, order_index.cuda())
        else:
            x = torch.index_select(a, 1, order_index)
        return F.pixel_shuffle(x, upscale_factor=self.upscale)


class Haar(nn.Module):
    def __init__(self, inchan):
        super(Haar, self).__init__()
        self.haar = nn.Sequential(
            HaarDown(inchan),
            HaarUp(inchan, False)
        )


    def forward(self, x):
        x = self.haar(x)
        return x


class Down(nn.Module):
    def __init__(self, inchan, outchan, circular=False):
        super(Down, self).__init__()
        conv = CircularDoubleConv if circular else DoubleConv2d
        self.d = nn.Sequential(
            nn.AvgPool2d(2),
            conv(inchan, outchan)
        )

    def forward(self, x):
        x = self.d(x)
        return x

class Up(nn.Module):
    def __init__(self, inchan, outchan, bilinear=True, circular=False):
        super(Up, self).__init__()

        if bilinear:
            # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up = AvgUp(2)
        else:
            self.up = nn.ConvTranspose2d(inchan//2, outchan//2, 2, stride=2)

        self.conv = CircularDoubleConv(inchan, outchan) if circular else DoubleConv2d(inchan, outchan)

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
        self.conv = nn.Conv2d(inchan, outchan, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class TightFrameUNet(nn.Module):
    def __init__(self, chan, residual=False, circular=False):
        super(TightFrameUNet, self).__init__()
        self.inc = CircularDoubleConv(chan, 64) if circular else DoubleConv2d(chan, 64)
        self.outc = OutConv(64, chan)
        self.down1 = Down(64, 128, circular=circular)
        self.down2 = Down(128, 256, circular=circular)
        self.down3 = Down(256, 512, circular=circular)
        self.down4 = Down(512, 512, circular=circular)
        self.haar1 = Haar(64)
        self.haar2 = Haar(128)
        self.haar3 = Haar(256)
        self.haar4 = Haar(512)
        self.haar5 = Haar(512)
        self.up1 = Up(2560, 256, circular=circular)
        self.up2 = Up(1280, 128, circular=circular)
        self.up3 = Up(640, 64, circular=circular)
        self.up4 = Up(320, 64, circular=circular)
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

        x = self.up1(x5, x4_haar, x4)
        x = self.up2(x, x3_haar, x3 )
        x = self.up3(x, x2_haar, x2)
        x = self.up4(x, x1_haar, x1)
        if self.residual:
            x = self.outc(x) + temp
        else:
            x = self.outc(x)
        return x


class TightFrameUNetSubbands(nn.Module):
    def __init__(self, chan, circular=False):
        super(TightFrameUNetSubbands, self).__init__()
        self.chan = chan
        if chan > 1:
            self.net1 = TightFrameUNet(chan//2, circular=circular)
            self.net2 = TightFrameUNet(chan//2, circular=circular)
        else:
            self.net = TightFrameUNet(1, circular=circular)

    def forward(self, x):
        if self.chan > 1:
            x1 = self.net1(x.narrow(1, 0, self.chan//2))
            x2 = self.net2(x.narrow(1, self.chan//2, self.chan//2))
            return torch.cat([x1, x2], dim=1)
        else:
            return self.net.forward(x)


class OverKillUNet(nn.Module):
    def __init__(self, chan):
        super(OverKillUNet, self).__init__()
        self.chan = chan
        # self.nets = [TightFrameUNet(1) for i in xrange(chan)]
        self.net0 = TightFrameUNet(1)
        self.net1 = TightFrameUNet(1)
        self.net2 = TightFrameUNet(1)
        self.net3 = TightFrameUNet(1)
        self.net4 = TightFrameUNet(1)
        self.net5 = TightFrameUNet(1)
        self.net6 = TightFrameUNet(1)
        self.net7 = TightFrameUNet(1)

    # def cuda(self, device=None):
    #     super(OverKillUNet, self).cuda(device)
        # for n in self.nets:
        #     n = n.cuda()
        # return self

    def forward(self, x):
        # X = [self.nets[i].forward(x.narrow(1, i, 1)) for i in xrange(self.chan)]
        x0 = self.net0.forward(x.narrow(1, 0, 1))
        x1 = self.net0.forward(x.narrow(1, 1, 1))
        x2 = self.net0.forward(x.narrow(1, 2, 1))
        x3 = self.net0.forward(x.narrow(1, 3, 1))
        x4 = self.net0.forward(x.narrow(1, 4, 1))
        x5 = self.net0.forward(x.narrow(1, 5, 1))
        x6 = self.net0.forward(x.narrow(1, 6, 1))
        x7 = self.net0.forward(x.narrow(1, 7, 1))
        return torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=1)


