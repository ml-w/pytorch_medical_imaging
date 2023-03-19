import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import *
from typing import Optional, Iterable
from .StandardLayers import PermuteTensor, _activation

__all__ = ['InvertedConv3d', 'Conv3d', 'DoubleConv3d', 'ConvTrans3d', 'ResidualBlock3d',
           'MultiConvResBlock3d']

activation_funcs = {
    'relu': nn.ReLU,
    'prelu': nn.PReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'tanh': nn.Tanh,
}

class InvertedConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kern_size=3, stride=1, padding=1, bias=True):
        super(InvertedConv3d, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, kern_size, stride, padding=padding, bias=bias)
        )

    def forward(self, x):
        return self.conv(x)


class Mask3d(nn.Module):
    r"""This class serves the sole purpose of masking the input 3d tensor that is zero padded along certain axis. This
    module was implemented to prevent propagation of bias from the padded slices to the slices with contents. Typical
    usage is to attach this mask after each operation with a global bias (e.g., Conv3d, BatchNorm3d, LayerNorm...etc).
    This class also assumes that the zero padding was always done at the tail.

    .. note::
        This class has no learnable parameters and will only mask the input tensor `x` based on the other parameters.

    """
    def __init__(self):
        super(Mask3d, self).__init__()

    def forward(self, x: torch.Tensor, seq_length: Optional[Tuple[int]] = None, axis: Optional[int]=-1) -> torch.Tensor:
        r"""Performs a masked 3D convolution on the input tensor `x`, using the specified convolution kernel.

        Args:
            x (torch.Tensor):
                The input tensor for the convolution operation, of shape (B x C x H x W x Z).
            seq_length (Optional[Tuple[int]]):
                A tuple of integers indicating the sequence length of the input tensor along the specified `axis`.
                If `seq_length` is not None, a mask will be applied to the convolution output, such that the convolution
                operation is not applied to certain elements of the input tensor. Default is `None`.
            axis (int):
                The axis along which the sequence elements are located in the input tensor. Default is -1, which is the
                last dimension

        Returns:
            torch.Tensor
        """
        # If seq_legnth is `None`, do nothing.
        if seq_length is None:
            return x
        else:
            # check length
            if not x.shape[0] == len(seq_length):
                raise ValueError(f"Batch size {x.shape} and the length of seq_length ({seq_length}) doesn't match.")

            # create a mask that is applied after the convolution
            axis = axis % x.dim() # handle negative values
            mask_size = [s if i in (0, axis) else 1 for i, s in enumerate(x.shape)]
            mask = torch.ones(mask_size, dtype=bool).to(x.device).expand_as(x)
            for i, l in enumerate(seq_length):
                ori_len = mask[i].shape[axis-1]
                mask[i].narrow(axis - 1, l, ori_len - l).fill_(0) # Mask
            return x * mask


class Conv3d(nn.Module):
    def __init__(self, in_ch, out_ch, kern_size=3, stride=1, padding=1, bias=True, activation='relu'):
        super(Conv3d, self).__init__()

        if not activation in activation_funcs:
            raise AttributeError(f"Activation should be one of [{'|'.join(activation_funcs.keys())}], "
                                 f"got {activation} instead")
        activation = activation_funcs.get(activation)

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kern_size, stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_ch),
            activation()
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv3d(nn.Module):
    r"""A 3D convolutional neural network block consisting of two consecutive convolutional layers followed by a 3D
    dropout layer.

    Args:
        in_ch (int):
            The number of input channels to the first convolutional layer.
        out_ch (int):
            The number of output channels from the second convolutional layer.
        kern_size (Union[int, Tuple[int, int, int]]):
            The size of the convolutional kernel(s) used in both convolutional layers. If an integer is specified, a
            cubic kernel of that size is used. Default is 3.
        stride (Union[int, Tuple[int, int, int]]):
            The stride(s) of the convolutional kernel(s) used in the first convolutional layer. If an integer is
            specified, the same stride is used in all dimensions. Default is 1.
        padding (Union[int, Tuple[int, int, int]]):
            The padding(s) for the convolutional kernel(s) used in both convolutional layers. If an integer is
            specified, the same padding is used in all dimensions. Default is 1.
        bias (bool):
            Whether or not to include a bias term in the convolutional layers.
            Default: True.
        dropout (float):
            The dropout probability for the 3D dropout layer. If zero, no dropout is applied. Default is 0.
        activation (str):
            The activation function to use after each convolutional layer. Must be one of the following: {'relu',
            'prelu', 'leaky_relu', 'elu', 'tanh'}. Default is 'relu'.
        mask (bool):
            Whether to use a :class:`MaskedConv3d` layer instead of a regular Conv3d layer for the convolutional layers.
            If true, the convolutional layers will perform masked convolutions. Default is `False`.
    """
    def __init__(self, in_ch, out_ch, kern_size=3, stride=1, padding=1, bias=True, dropout=0, activation='relu',
                 mask=False):
        super(DoubleConv3d, self).__init__()
        self.conv = nn.Sequential(
            Conv3d(in_ch, out_ch, kern_size=kern_size, stride=stride, padding=padding, bias=bias, activation=activation),
            Conv3d(out_ch, out_ch, kern_size=kern_size, padding=padding, bias=bias, activation=activation),
            nn.Dropout3d(p = dropout, inplace=False)
        )

    def forward(self, x):
        return self.conv(x)

class ConvTrans3d(nn.Module):
    def __init__(self, in_ch, out_ch, kern_size=3, stride=1, padding=1, bias=True, activation='relu'):
        super(ConvTrans3d, self).__init__()

        if not activation in activation_funcs:
            raise AttributeError(f"Activation should be one of [{'|'.join(activation_funcs.keys())}], "
                                 f"got {activation} instead")
        activation = activation_funcs.get(activation)

        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kern_size, stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_ch),
            activation(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.2):
        super(ResidualBlock3d, self).__init__()
        self._in_ch = in_ch
        self._out_ch = out_ch

        self.in_conv = nn.Sequential(
            Conv3d(in_ch, out_ch // 4, 1, 1, bias=False, padding=0),
            Conv3d(out_ch // 4, out_ch // 4, kern_size=3, bias=False, padding=1)
        )
        self.in_bn = nn.BatchNorm3d(in_ch)

        self.out_conv = nn.Conv3d(out_ch // 4, out_ch, 1, 1, bias=False, padding=0)
        self.pre_add_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.dropout = nn.Dropout3d(p=p)

    def forward(self, x):
        while x.dim() < 5:
            x = x.unsqueeze(0)
        res = x
        pre_out = F.relu(self.in_bn(x), inplace=True)
        out = self.dropout(self.in_conv(pre_out))
        out = self.out_conv(out)

        if (self._in_ch != self._out_ch):
            res = self.pre_add_conv(pre_out)

        out += res
        return out


class MultiConvResBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch, num_of_convs, kern_size=5, padding=2, drop_out=0, bias=True, activation='relu'):
        assert num_of_convs > 1, "Number of convolutions must be larger than 1."
        super(MultiConvResBlock3d, self).__init__()
        self.first_conv = Conv3d(in_ch, out_ch, kern_size=kern_size, padding=padding, bias=bias, activation=activation)
        self.convs = [Conv3d(out_ch,out_ch, kern_size=kern_size, padding=padding, bias=bias, activation=activation) for i in range(num_of_convs - 1)]
        self.conv = nn.Sequential(*self.convs)
        self.dropout = nn.Dropout3d(p=drop_out)

    def forward(self, x):
        x = self.first_conv(x)
        return self.dropout(self.conv(x)) + x




