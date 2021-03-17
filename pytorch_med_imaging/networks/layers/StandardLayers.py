import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DoubleConv1d','DoubleConv2d', 'LinearDoubleConv', 'CircularDoubleConv', 'ReflectiveDoubleConv',
           'PermuteTensor',
           'StandardFC', 'StandardFC2d']

_activation = {
    'relu': nn.ReLU,
    'leaky-relu': nn.LeakyReLU,
    'relu6': nn.ReLU6,
    'tanh': nn.Tanh
}


class DoubleConv1d(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, padding=1, **kwargs):
        super(DoubleConv1d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1, **kwargs),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, **kwargs),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class DoubleConv2d(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, padding= 1, **kwargs):
        super(DoubleConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, **kwargs),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, **kwargs),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class LinearDoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, padding=1, **kwargs):
        super(LinearDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=padding, **kwargs),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=padding, **kwargs),
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
    def __init__(self, permute_order: list or tuple or torch.Tensor):
        super(PermuteTensor, self).__init__()
        if isinstance(permute_order, list) or isinstance(permute_order, tuple):
            _permute_order = torch.Tensor(permute_order).int()
        elif isinstance(permute_order, torch.Tensor):
            _permute_order = permute_order.int()
        else:
            raise TypeError("Input should be a list, tuple or torch.Tensor, "
                            "got {} instead.".format(type(permute_order)))

        self.register_buffer('_permute_order', _permute_order, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self._permute_order.tolist())


class StandardFC(nn.Module):
    def __init__(self, in_ch, out_ch, activation='relu', dropout=0.2):
        super(StandardFC, self).__init__()


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


class StandardFC2d(nn.Module):
    r"""
    Standard FC for $(B x S x C)$ inputs where S is an arbitrary dimension such that each of the elements in
    the batch hold S extra set of C-lengthed 1d vector features.

    Args:
        in_ch (int):
            Number of input channels, should equal the length of last dimension of expected input.
        out_ch (int):
            Number of desired output channels.
        activation (str, Optional):
            Which activation to use. Default to `relu`.
        dropout (float, Optional):
            Drop out probability. Default to 0.2
    """
    def __init__(self, in_ch, out_ch, activation='relu', dropout=0.2):
        super(StandardFC2d, self).__init__()

        if not activation in _activation:
            raise AttributeError("Activation layer requested ({})is not in list. Available activations are:"
                                 "{}".format(activation, list(_activation.keys())))
        self.activation = _activation[activation]

        self._fc = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            PermuteTensor([0, 2, 1]),
            nn.BatchNorm1d(out_ch),
            PermuteTensor([0, 2, 1]),
            self.activation(),
            nn.Dropout(p = dropout)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.dim() == 3:
            raise ArithmeticError("Expect dim to be 3, got {} instead.".format(x.dim()))
        return self._fc(x)

