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
