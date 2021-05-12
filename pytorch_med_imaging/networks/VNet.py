import torch
import torch.nn as nn
from .layers import Conv3d, ConvTrans3d, MultiConvResBlock3d, DoubleConv3d
from numpy import clip

__all__ = ['VNet']


class Down(nn.Module):
    r"""Down and than convolution."""
    def __init__(self,
                 in_ch,
                 out_ch,
                 num_conv: int = 2,
                 down_mode: str = 'conv',
                 drop_out: float = 0.2,
                 residual: bool = True):
        super(Down, self).__init__()
        assert  num_conv >= 1, "Must have at least 1 convolution."
        self.residual = residual

        if down_mode == 'conv':
            self.add_module('down', nn.Conv3d(in_ch, out_ch, stride=2, kernel_size=2, padding=0))
        elif down_mode == 'pool':
            self.add_module('down', nn.MaxPool3d(2))
        else:
            raise AttributeError(f"Argument down_mode can only be one of ['conv'|'poo'], got {down_mode} instead")

        self.conv = MultiConvResBlock3d(out_ch, out_ch, num_conv, drop_out=drop_out)

    def forward(self, x):
        x = self.down(x)
        if self.residual:
            return self.conv(x).add(x)
        else:
            return self.conv(x)



class Up(nn.Module):
    def __init__(self, in_ch, num_of_convs, drop_out=0):
        r"""Upwards transition. First up, than conv."""
        super(Up, self).__init__()
        self.conv = MultiConvResBlock3d(in_ch, in_ch // 2, num_of_convs, drop_out=drop_out)
        self.up = ConvTrans3d(in_ch, in_ch //2, kern_size=2, stride=2, padding=0)

    def forward(self, x, s):
        x = self.up(x)
        return self.conv(torch.cat([x, s], dim=1)).mul(x)

class VNet(nn.Module):
    def __init__(self, in_ch, out_ch, init_conv_out_ch=16, depth=5, residual=True):
        super(VNet, self).__init__()
        self.residual = True
        self.depth = depth
        start_ch = init_conv_out_ch

        # First conv
        self.add_module('in_conv', DoubleConv3d(in_ch, start_ch))

        # Down
        self.down = nn.ModuleDict()
        for d in range(depth - 1): #[0, 1, 2]
            num_conv = 2 if d == 1 else 3
            self.down.add_module(f'down_{d+1:02d}', Down(start_ch * 2 **(d), start_ch * 2 **(d + 1), num_conv))

        # Last down
        self.add_module('last_conv',
                        MultiConvResBlock3d(start_ch * 2 ** (depth - 2), start_ch * 2 ** (depth - 1), 2))

        # Up
        self.up = nn.ModuleDict()
        for d in range(depth-1, 1, -1): #[4, 3, 2]
            num_conv = 2 if d == 1 else 3
            layer = Up(start_ch * 2 ** d, clip(d+1, 2, 3))
            self.up.add_module(f'up_{d+1:02d}', layer)

        # Last up
        self.up.add_module('last_up', Up(start_ch * 2, start_ch))

        # Out conv
        self.out_conv = nn.Conv3d(start_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.in_conv(x) + x

        short_cut = [x]
        for layer in self.down:
            x = self.down[layer](x)
            # no need to append last:
            if len(short_cut) < self.depth - 1:
                short_cut.append(x)

        for i, layer in enumerate(self.up):
            d = self.depth - i - 2
            s = short_cut[d]
            x = self.up[layer](x, s)

        return self.out_conv(x)