import torch
import torch.nn as nn
from .Layers import Conv3d, ConvTrans3d, MultiConvResBlock3d
from numpy import clip


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, down_fact=2):
        super(Down, self).__init__()
        self.down_conv = Conv3d(in_ch, out_ch, kern_size=down_fact, stride=down_fact, padding=0)

    def forward(self, x):
        return x, self.down_conv(x) # x is shortcut connection

class Up(nn.Module):
    def __init__(self, in_ch, num_of_convs, drop_out=0, up_fact=2):
        super(Up, self).__init__()
        self.conv = MultiConvResBlock3d(in_ch, in_ch // 2, num_of_convs, drop_out=drop_out)
        self.up = ConvTrans3d(in_ch // 2, in_ch //2, kern_size=up_fact, stride=up_fact, padding=0)

    def forward(self, x, s):
        return self.up(self.conv(torch.cat([x, s], dim=1)))

class VNet(nn.Module):
    def __init__(self, in_ch, out_ch, depth=5):
        super(VNet, self).__init__()
        self.depth=5
        start_ch = 16

        # First Down
        self.down = [Down(in_ch, start_ch)]

        # Inner down
        for d in range(depth-2): #[0, 1, 2]
            layer = nn.Sequential(
                MultiConvResBlock3d(start_ch * 2 ** d, start_ch * 2 ** (d+1), clip(d+1, 2, 3)),
                Down(start_ch * 2 ** (d+1), start_ch * 2 ** (d+1))
            )
            self.down.append(layer)

        # Last down
        self.last_down = MultiConvResBlock3d(start_ch * 2 ** (depth - 2), start_ch * 2 ** (depth - 1))

        # First up
        self.up = [ConvTrans3d(start_ch * 2 **(depth-1), start_ch * 2 ** (depth-1), kern_size=2, stride=2, padding=0)]

        # Inner up
        for d in range(depth-1, 1, -1): #[4, 3, 2]
            layer = Up(start_ch * 2 ** d, start_ch * 2 ** (d-1), clip(d+1, 2, 3))
            self.up.append(layer)

        # Last up
        self.up.append(nn.Sequential(
            Up(32, 2),
            Conv3d(16, out_ch, kern_size=1, padding=0)
        ))

    def forward(self, x):
        short_cut = []
        for layer in self.down:
            s, x = layer(x)
            short_cut.append(x)

        x = self.last_down(x)
        for i, layer in enumerate(self.up):
            d = self.depth - i - 1
            s = short_cut[d]
            x = layer(x, s)

        return x