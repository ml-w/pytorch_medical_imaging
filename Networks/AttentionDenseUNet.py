from .Layers import AttentionBlock, AttentionGating
from .DenseUNet import DenseBlock, DenseConv, Transition, Down, Up

import torch.nn as nn

class AttentionDenseUNet2D(nn.Module):
    def __init__(self, n_channels, n_classes, gen_attmap=False):
        super(AttentionDenseUNet2D, self).__init__()
        self.attmap = gen_attmap
        self.att = None

        k = 32
        init_conv_out = 96
        self.inconv = DenseConv(n_channels, init_conv_out, 7, stride=2, pad=True)
        self.pooling = nn.MaxPool2d(3, stride=2, padding=1)
        self.down1 = DenseBlock(init_conv_out, k, 6)
        self.down2 = Down(init_conv_out + 6 * k, k, 12)
        self.down3 = Down(init_conv_out + 18* k, k, 36)
        self.down4 = Down(init_conv_out + 54* k, k, 24)

        self.up1 = Up(init_conv_out * 2 + 132 * k, 768)
        self.up2 = Up(768 + init_conv_out + 18* k, 384)
        self.up3 = Up(384 + init_conv_out + 6 * k, 96)
        self.up4 = Up(96  + init_conv_out, 96)
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, align_corners=True, mode='bilinear'),
            DenseConv(96, 64)
        )
        self.lastup = DenseConv(64, n_classes, 1)

        self.gating = AttentionGating(init_conv_out + 54* k, 256)
        self.att1 = AttentionBlock(768, 256)
        self.att2 = AttentionBlock(384, 256)
        self.att3 = AttentionBlock(96, 256)
        self.att4 = AttentionBlock(96, 256)
        self.att5 = AttentionBlock(64, 256)

    def forward(self, x):
        d1 = self.inconv(x)
        # print d1.shape
        d2 = self.down1(self.pooling(d1))
        # print d2.shape
        d3 = self.down2(d2)
        # print d3.shape
        d4 = self.down3(d3)
        # print d4.shape
        d5 = self.down4(d4)
        # print d5.shape
        g = self.gating(d4)

        x = self.up1(d5, d4)
        x, att1 = self.att1(x, g)
        # print x.shape
        x = self.up2(x, d3)
        x, att2 = self.att2(x, g)
        # print x.shape
        x = self.up3(x, d2)
        x, att3 = self.att3(x, g)
        # print x.shape
        x = self.up4(x, d1)
        x, att4 = self.att4(x, g)
        # print x.shape
        x = self.up5(x)
        x, att5 = self.att5(x, g)
        # print x.shape
        x = self.lastup(x)
        # print x.shape

        if self.attmap:
            self.att = [att1, att2, att3, att4, att5]
            return x
        else:
            del att1, att2, att3, att4, att5
            return x

    def get_att_map(self):
        return self.att