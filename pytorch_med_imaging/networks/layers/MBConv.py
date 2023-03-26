import torch
import torch.nn as nn
from typing import Optional
from einops.layers.torch import Rearrange, Reduce

# MBConv25d
class SqueezeExcitation25d(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w z -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.rand((x.shape[0], 1, 1, 1, 1)).uniform_().to(device) > self.prob
        return x * keep_mask / (1 - self.prob)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class MBConv25d(nn.Module):
    r"""MBConv implementation. Expect input size `(B × in_ch × H × W × D)`. The drop out is special and drops the
    attension from a patch as a whole rather than drop random features across the whole batch.

    Args:
        in_ch (int):
            Number of input channels.
        out_ch (int):
            Number of output channels.
        downsample (bool):
            If true, downsamples the input by using a convolution stride of `[2, 2, 1]`.
        expansion_rate (float):
            Expansion rate for the hidden dimension of the block. Default is 4.
        shrinkage_rate (float):
            Shrinkage rate for the squeeze and excitation layer. Default is 0.25.
        dropout (float):
            Dropout probability. Default is 0.

    Attributes:
        mbconv (nn.Sequential): Sequence of convolutional layers and squeeze and excitation layers.

    Example:
        >>> block = MBConv25d(in_ch=64, out_ch=128, downsample=True)
        >>> x = torch.randn(1, 64, 32, 32, 8)
        >>> out = block(x)
    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 downsample: Optional[bool] = False,
                 expansion_rate: Optional[int] = 4,
                 shrinkage_rate: Optional[float] = 0.25,
                 dropout: Optional[float] = 0.2) -> None:
        super().__init__()
        hidden_dim = int(expansion_rate * out_ch)
        stride = 2 if downsample else 1

        self.mbconv = nn.Sequential(
            nn.Conv3d(in_ch, hidden_dim, 1),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, [3, 3, 1], stride = [stride, stride, 1], padding = [1, 1, 0], groups = hidden_dim),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            SqueezeExcitation25d(hidden_dim, shrinkage_rate = shrinkage_rate),
            nn.Conv3d(hidden_dim, out_ch, 1),
            nn.BatchNorm3d(out_ch)
        )

        if in_ch == out_ch and not downsample:
            self.mbconv = MBConvResidual(self.mbconv, dropout = dropout)

    def forward(self, x):
        return self.mbconv(x)