import torch
import torch.nn as nn

__all__ = ['DenseConv', 'DenseConv3D','DenseLayer', 'DenseBlock', 'DenseBlock3D', 'DenseLayer3D']

class DenseConv(nn.Module):
    def __init__(self, inchan, outchan, kernsize=3, stride=1, pad=True):
        super(DenseConv, self).__init__()
        padding = int((kernsize - 1)/2.) if pad else 0
        self.conv = nn.Sequential(
            nn.BatchNorm2d(inchan),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchan, outchan, kernsize, padding=padding, stride=stride, bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class DenseLayer(nn.Module):
    def __init__(self, inchan, k, bn_size):
        super(DenseLayer, self).__init__()

        self.convs = nn.Sequential(
            DenseConv(inchan, bn_size * k, kernsize=1),
            DenseConv(bn_size * k, k)
        )

    def forward(self, x):
        new_features = self.convs(x)
        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, inchan, k, num_layers):
        super(DenseBlock, self).__init__()

        inconv = DenseLayer(inchan, k, 4)

        convs = [inconv]
        for i in range(num_layers - 1):
            convs.append(
                DenseLayer(inchan + (i+1)*k, k, 4)
            )
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)


class DenseConv3D(nn.Module):
    def __init__(self, inchan, outchan, kernsize=3, stride=1, pad=True):
        super(DenseConv3D, self).__init__()
        padding = int((kernsize - 1)/2.) if pad else 0
        self.conv = nn.Sequential(
            nn.BatchNorm3d(inchan),
            nn.ReLU(inplace=True),
            nn.Conv3d(inchan, outchan, kernsize, padding=padding, stride=stride, bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class DenseLayer3D(nn.Module):
    def __init__(self, inchan, k, bn_size, kernsize=3, dropout=0.2):
        super(DenseLayer3D, self).__init__()

        self.convs = nn.Sequential(
            DenseConv3D(inchan, bn_size * k, kernsize=1),
            DenseConv3D(bn_size * k, k, kernsize=kernsize),
            nn.Dropout3d(p=dropout)
        )

    def forward(self, x):
        new_features = self.convs(x)
        return torch.cat([x, new_features], dim=1)


class DenseBlock3D(nn.Module):
    def __init__(self, inchan:int, k:int, num_layers:int , kernsize: int or [int] = 3, dropout:float =0.2):
        super(DenseBlock3D, self).__init__()

        convs = []
        for i in range(num_layers):
            convs.append(
                DenseLayer3D(inchan + (i+1)*k, k, 4, dropout=dropout, kernsize=kernsize)
            )
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)
