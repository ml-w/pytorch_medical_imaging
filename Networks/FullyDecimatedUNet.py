import torch
import torch.nn as nn
import torch.nn.functional as F

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


class LinearDoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(LinearDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class CircularDoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, linear=False):
        super(CircularDoubleConv, self).__init__()
        if linear:
            self.conv0 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3),
                nn.BatchNorm2d(out_ch)
            )
            self.conv1 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.conv0 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.conv1 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )


    @staticmethod
    def pad_circular(x, pad, dim=[-2, -1]):
        """

        :param x: shape [H, W]
        :param pad: int >= 0
        :return:
        """
        indim = x.dim()

        indexes_1 = []
        indexes_2 = []
        indexes_3 = []
        indexes_4 = []
        for i in xrange(indim):
            if i == dim[0] % indim:
                indexes_1.append(slice(0, pad))
                indexes_2.append(slice(None))
                indexes_3.append(slice(-2*pad, -pad))
                indexes_4.append(slice(None))
            elif i == dim[1] % indim:
                indexes_1.append(slice(None))
                indexes_2.append(slice(0, pad))
                indexes_3.append(slice(None))
                indexes_4.append(slice(-2*pad, -pad))
            else:
                indexes_1.append(slice(None))
                indexes_2.append(slice(None))
                indexes_3.append(slice(None))
                indexes_4.append(slice(None))

        x = torch.cat([x, x[indexes_1]], dim=dim[0])
        x = torch.cat([x, x[indexes_2]], dim=dim[1])
        x = torch.cat([x[indexes_3], x], dim=dim[0])
        x = torch.cat([x[indexes_4], x], dim=dim[1])
        return x

    def forward(self, x):
        x = self.pad_circular(x, 1)
        x = self.conv0(x)
        x = self.pad_circular(x, 1)
        x = self.conv1(x)
        return x

    def forward(self, x):
        x = self.pad_circular(x, 1)
        x = self.conv0(x)
        x = self.pad_circular(x, 1)
        x = self.conv1(x)
        return x

class Decimation(nn.Module):
    def __init__(self, inchan):
        super(Decimation, self).__init__()
        self.kern_matrix = torch.tensor([[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]]).view(4, 2, 2)
        self.weights = torch.zeros([inchan*4, inchan, 2, 2])
        for i in xrange(inchan):
            self.weights[4*i:4*i+4, i] = self.kern_matrix

    def forward(self, x):
        if x.is_cuda:
            x = F.conv2d(x, self.weights.cuda(), stride=2)
        else:
            x = F.conv2d(x, self.weights, stride=2)
        return x


class Interpolation(nn.Module):
    def __init__(self, inchan):
        super(Interpolation, self).__init__()
        self.kern_matrix = torch.tensor([[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]]).view(4, 2, 2)
        self.weights = torch.zeros([inchan, inchan/4, 2, 2])
        for i in xrange(inchan/4):
            self.weights[4*i:4*i+4, i] = self.kern_matrix

    def forward(self, x):
        if x.is_cuda:
            x = F.conv_transpose2d(x, self.weights.cuda(), stride=2)
        else:
            x = F.conv_transpose2d(x, self.weights, stride=2)
        return x

class Down(nn.Module):
    def __init__(self, inchan, outchan, linear=False):
        super(Down, self).__init__()
        self.decimation = Decimation(inchan)

        if not linear:
            conv = DoubleConv
        else:
            conv = LinearDoubleConv

        self.conv0 = conv(inchan, outchan // 4)
        self.conv1 = conv(inchan, outchan // 4)
        self.conv2 = conv(inchan, outchan // 4)
        self.conv3 = conv(inchan, outchan // 4)

        self.inchan = inchan

    def forward(self, x):
        # B x C x H x W --> B x outchan x H/2 x W/2
        dec = self.decimation(x)
        x1 = self.conv0(dec.narrow(1, 0, self.inchan))
        x2 = self.conv1(dec.narrow(1, self.inchan, self.inchan))
        x3 = self.conv2(dec.narrow(1, self.inchan * 2, self.inchan))
        x4 = self.conv3(dec.narrow(1, self.inchan * 3, self.inchan))
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return dec, out

class LastDown(nn.Module):
    def __init__(self, inchan, outchan, linear=False):
        super(LastDown, self).__init__()
        self.decimation = Decimation(inchan)

        conv = CircularDoubleConv
        self.conv0 = conv(inchan, outchan // 4, linear=linear)
        self.conv1 = conv(inchan, outchan // 4, linear=linear)
        self.conv2 = conv(inchan, outchan // 4, linear=linear)
        self.conv3 = conv(inchan, outchan // 4, linear=linear)

        self.inchan = inchan

    def forward(self, x):
        # B x C x H x W --> B x outchan x H/2 x W/2
        dec = self.decimation(x)
        x1 = self.conv0(dec.narrow(1, 0, self.inchan))
        x2 = self.conv1(dec.narrow(1, self.inchan, self.inchan))
        x3 = self.conv2(dec.narrow(1, self.inchan * 2, self.inchan))
        x4 = self.conv3(dec.narrow(1, self.inchan * 3, self.inchan))
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return out

class Up(nn.Module):
    def __init__(self, inchan, outchan, linear=False):
        super(Up, self).__init__()

        assert inchan % 4 == 0
        self.up = nn.Upsample(scale_factor=2, align_corners=True, mode='bilinear')

        if not linear:
            conv = DoubleConv
        else:
            conv = LinearDoubleConv
        self.conv = conv(inchan, outchan)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        x = torch.cat([x1, x2], 1)
        x = self.conv(x)
        return x

class LastUp(nn.Module):
    def __init__(self, inchan, outchan, linear=False):
        super(LastUp, self).__init__()

        assert inchan % 4 == 0
        self.up = Interpolation(inchan)
        self.conv = CircularDoubleConv(inchan // 4, outchan, linear=linear)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class DUnet(nn.Module):
    def __init__(self, inchan):
        super(DUnet, self).__init__()
        self.indown = CircularDoubleConv(inchan, 64, linear=True)
        self.down1 = LastDown(64, 256, linear=True)
        self.down2 = LastDown(256, 1024, linear=True)
        self.down3 = LastDown(1024, 1024, linear=True)
        self.up1 = Up(2048, 256, linear=True)
        self.up2 = Up(512, 64, linear=True)
        self.up3 = Up(128, 16, linear=True)
        self.outup = CircularDoubleConv(16, inchan, linear=True)


    def forward(self, x):
        x0 = self.indown(x)     # 32
        x1 = self.down1(x0)     # 128
        x2 = self.down2(x1)     # 512
        x3 = self.down3(x2)     # 2048

        x = self.up1(x2, x3)    # 2048 + 512 -> 512
        x = self.up2(x1, x)     # 512 + 128 -> 128
        x = self.up3(x0, x)
        x = self.outup(x)
        return x

class LinearDUnet(nn.Module):
    def __init__(self, inchan):
        super(LinearDUnet, self).__init__()
        self.indown = LinearDoubleConv(inchan, 64)
        self.down1 = LastDown(64, 256, linear=True)
        self.down2 = LastDown(256, 1024, linear=True)
        self.down3 = LastDown(1024, 1024, linear=True)
        self.up1 = Up(2048, 256, linear=True)
        self.up2 = Up(512, 64, linear=True)
        self.up3 = Up(128, 16, linear=True)
        self.outup = LinearDoubleConv(16, inchan)


    def forward(self, x):
        x0 = self.indown(x)     # 32
        x1 = self.down1(x0)     # 128
        x2 = self.down2(x1)     # 512
        x3 = self.down3(x2)     # 2048

        x = self.up1(x2, x3)    # 2048 + 512 -> 512
        x = self.up2(x1, x)     # 512 + 128 -> 128
        x = self.up3(x0, x)
        x = self.outup(x)
        return x

class FullyDecimatedUNet(nn.Module):
    def __init__(self, inchan):
        super(FullyDecimatedUNet, self).__init__()
        self.inchan = inchan
        if inchan > 1:
            self.unet1 = DUnet(inchan // 2)
            self.unet2 = DUnet(inchan // 2)
        else:
            self.unet = DUnet(1)

    def forward(self, x):
        if self.inchan > 1:
            x1 = self.unet1(x.narrow(1, 0, self.inchan //2))
            x2 = self.unet2(x.narrow(1, self.inchan //2 , self.inchan //2))
            return torch.cat([x1, x2],1)
        else:
            return self.unet.forward(x)


class FullyDecimatedUNet_Single(nn.Module):
    def __init__(self):
        super(FullyDecimatedUNet_Single, self).__init__()
        self.dunet = DUnet(1)

    def forward(self, x):
        x = self.dunet.forward(x)
        return x

class PRDecimateUNet(nn.Module):
    def __init__(self, inchan):
        super(PRDecimateUNet, self).__init__()

        self.inchan = inchan
        self.net = LinearDUnet(1)

    def forward(self, x):
        out = [self.net.forward(x.narrow(1, i, 1)) for i in xrange(self.inchan//2)] + \
              [self.net.forward(x.narrow(1, i, 1).transpose(-1, -2)).transpose(-1, -2) for i in xrange(self.inchan//2, self.inchan)]

        return torch.cat(out, 1)
