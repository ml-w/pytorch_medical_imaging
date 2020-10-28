import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DoubleConv', 'LinearDoubleConv', 'CircularDoubleConv', 'ReflectiveDoubleConv', 'PermuteTensor',
           'FC']

_activation = {
    'relu': nn.ReLU,
    'leaky-relu': nn.LeakyReLU,
    'relu6': nn.ReLU6,
    'tanh': nn.Tanh
}

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
        for i in range(indim):
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


class ReflectiveDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernsize=3, linear=False):
        super(ReflectiveDoubleConv, self).__init__()

        pad = kernsize // 2

        if linear:
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_ch, out_ch, kernsize),
                nn.BatchNorm2d(out_ch),
                nn.ReflectionPad2d(pad),
                nn.Conv2d(out_ch, out_ch, kernsize),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_ch, out_ch, kernsize),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(pad),
                nn.Conv2d(out_ch, out_ch, kernsize),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )


    def forward(self, x):
        return self.conv(x)


class PermuteTensor(nn.Module):
    def __init__(self, permute_order):
        super(PermuteTensor, self).__init__()
        if not isinstance(permute_order, torch.Tensor):
            permute_order = torch.Tensor(permute_order)

        self.register_buffer('permute_order', permute_order.int(), True)

    def forward(self, x):
        return x.permute(*self.permute_order.tolist())


class FC(nn.Module):
    def __init__(self, in_ch, out_ch, activation='relu', dropout=0.2):
        super(FC, self).__init__()


        if not activation in _activation:
            raise AttributeError("Activation layer requested ({})is not in list. Available activations are:"
                                 "{}".format(activation, list(_activation.keys())))

        self._activation = _activation[activation]

        self._fc = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            self._activation(),
            nn.Dropout(p = dropout)
        )

    def forward(self, x):
        return self._fc(x)
