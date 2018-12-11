import torch
import torch.nn as nn

from Layers import DoubleConv, AttentionBlock, AttentionGating
from .UNet import up, down, outconv, inconv

class AttentionUNet(nn.Module):
    def __init__(self, n_channels, outchan=None, gen_attmap=False):
        super(AttentionUNet, self).__init__()
        self.attmap = gen_attmap
        self.att = None

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, True)
        self.up2 = up(512, 128, True)
        self.up3 = up(256, 64, True)
        self.up4 = up(128, 64, True)
        self.outc = outconv(64, n_channels) if outchan is None else outconv(64, outchan)

        self.gating = AttentionGating(512, 256)

        self.att1 = AttentionBlock(256, 256)
        self.att2 = AttentionBlock(128, 256)
        self.att3 = AttentionBlock(64, 256)
        self.att4 = AttentionBlock(64, 256)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        g = self.gating(x4)

        u1 = self.up1(x5, x4)
        u1, att1 = self.att1(u1, g)

        u2 = self.up2(u1, x3)
        u2, att2 = self.att2(u2, g)

        u3 = self.up3(u2, x2)
        u3, att3 = self.att3(u3, g)

        u4 = self.up4(u3, x1)
        u4, att4 = self.att4(u4, g)

        u5 = self.outc(u4)

        if not self.attmap:
            del att1, att2, att3, att4
            self.att = None
            return u5
        else:
            self.att = [att1, att2, att3, att4]
            return u5

    def get_att_map(self):
        return self.att

