import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.StandardLayers3D import MaskedSequential3d
from .layers import ResidualBlock3d, Conv3d, InvertedConv3d, BGRUStack, BGRUCell, DoubleConv3d
from .layers.dummy_layers import SupportMask3d
from typing import Optional


class SoftMaskBranch_aed2d(nn.Module):
    def __init__(self, in_ch, out_ch, r=1):
        super(SoftMaskBranch_aed2d, self).__init__()

        down1 = [ResidualBlock3d(in_ch, out_ch) for i in range(r)]
        self.down1 = nn.Sequential(*down1)

        down2 = [ResidualBlock3d(out_ch, out_ch) for i in range(2 * r)]
        self.down2 = nn.Sequential(*down2)

        self.skip = ResidualBlock3d(out_ch, out_ch)

        up1 = [ResidualBlock3d(out_ch, out_ch) for i in range(r)]
        self.up1 = nn.Sequential(*up1)

        self.out_conv = nn.Sequential(
            InvertedConv3d(out_ch, out_ch, kern_size=1, padding=0),
            InvertedConv3d(out_ch, out_ch, kern_size=1, padding=0)
        )

        pass

    def forward(self, x):
        mask1 = F.max_pool3d(x, [3, 3, 1], stride=[2, 2, 1])
        mask1 = self.down1(mask1)

        mask2 = F.max_pool3d(mask1, [3, 3, 1], stride=[2, 2, 1])
        mask2 = self.down2(mask2)

        skip = self.skip(mask1)
        mask2 = F.interpolate(mask2, skip.size()[-3:])

        mask3 = self.up1(mask2 + skip)
        mask3 = F.interpolate(mask3, x.size()[-3:])

        out = self.out_conv(mask3)
        out = torch.sigmoid(out)
        return out


class SoftMaskBranch_25d(nn.Module, SupportMask3d):
    def __init__(self, in_ch, out_ch, r = 1, stage=0, mask=False):
        super(SoftMaskBranch_25d, self).__init__()

        self._in_ch = in_ch
        self._out_ch = out_ch
        self._stage = stage
        self._mask = mask

        if self._stage == 0:
            skipconnection = 3
        elif self._stage == 1:
            skipconnection = 2
        else:
            skipconnection = 1

        # Define kick start layers
        seq = nn.Sequential if not mask else MaskedSequential3d
        self.init_maxpool = nn.MaxPool3d(kernel_size=[3, 3, 1], stride=[2, 2, 1], padding=[1, 1, 0])
        self.in_res = seq(*[ResidualBlock3d(in_ch, out_ch, mask=mask)])
        self.skip0 = seq(*(
            [ResidualBlock3d(in_ch , out_ch, mask=mask)] +
            [ResidualBlock3d(out_ch, out_ch, mask=mask) for i in range(r - 1)]
        ))

        # Define softmask branch encoder and decoders
        for i in range(skipconnection):
            if not i == 0:
                self.add_module(f'skip{i}', seq(*[
                    ResidualBlock3d(out_ch, out_ch, mask=mask) for j in range(r)
                ]))
            self.add_module(f'down{i}', seq(*(
                [nn.MaxPool3d(kernel_size=[3, 3, 1], stride=[2, 2, 1], padding=[1, 1, 0])] +
                [ResidualBlock3d(out_ch, out_ch, mask=mask) for j in range(r * 2 if i == skipconnection - 1 else r)]
                # Last layer has 2r res blocks
            )))
            self.add_module(f'upres{i}', seq(*[
                ResidualBlock3d(out_ch, out_ch, mask=mask) for j in range(r)
            ]))

        # Define output layer
        self.output_res = seq(
            InvertedConv3d(out_ch, out_ch, kern_size=1, padding=0, mask=mask),
            InvertedConv3d(out_ch, out_ch, kern_size=1, padding=0, mask=mask),
            nn.Sigmoid()
        )

    def forward(self, x, seq_length=None, axis=-1):
        if not self._mask:
            skip0 = self.skip0(self.init_maxpool(x))
            dx = self.in_res(skip0)
            d0 = self.down0(skip0)
            if self._stage in (0, 1):
                d1, skip1 = self.down1(d0), self.skip1(d0)
                if self._stage == 0:
                    d2, skip2 = self.down2(d1), self.skip2(d1)
                    u2 = self.upres2(F.interpolate(d2, skip2.shape[-3:]) + skip2 + d1)
                else:
                    u2 = d1
                u1 = self.upres1(F.interpolate(u2, skip1.shape[-3:]) + skip1 + d0)
            else:
                u1 = d0
            u0 = self.upres0(F.interpolate(u1, skip0.shape[-3:]) + skip0 + dx)
            u0 = F.interpolate(u0, x.shape[-3:])
            u0 = self.output_res(u0)
            return u0
        else:
            skip0 = self.skip0(self.init_maxpool(x), seq_length=seq_length, axis=axis)
            dx = self.in_res(skip0, seq_length=seq_length, axis=axis)
            d0 = self.down0(skip0, seq_length=seq_length, axis=axis)
            if self._stage in (0, 1):
                d1, skip1 = self.down1(d0, seq_length=seq_length, axis=axis), \
                            self.skip1(d0, seq_length=seq_length, axis=axis)
                if self._stage == 0:
                    d2, skip2 = self.down2(d1, seq_length=seq_length, axis=axis), \
                                self.skip2(d1, seq_length=seq_length, axis=axis)
                    u2 = self.upres2(F.interpolate(d2, skip2.shape[-3:]) + skip2 + d1,
                                     seq_length=seq_length, axis=axis)
                else:
                    u2 = d1
                u1 = self.upres1(F.interpolate(u2, skip1.shape[-3:]) + skip1 + d0,
                                 seq_length=seq_length, axis=axis)
            else:
                u1 = d0
            u0 = self.upres0(F.interpolate(u1, skip0.shape[-3:]) + skip0 + dx,
                             seq_length=seq_length, axis=axis)
            u0 = F.interpolate(u0, x.shape[-3:])
            u0 = self.output_res(u0, seq_length=seq_length, axis=axis)
            return u0


class AttentionModule_25d(nn.Module, SupportMask3d):
    def __init__(self, in_ch, out_ch, stage, p=1, t=2, r=1, save_mask=False, mask=False):
        super(AttentionModule_25d, self).__init__()
        self._mask = mask
        seq = nn.Sequential if not mask else MaskedSequential3d
        in_conv = [ResidualBlock3d(in_ch, out_ch, mask=mask) for i in range(p)]
        self.in_conv = seq(*in_conv)

        self.soft_mask_branch = SoftMaskBranch_25d(out_ch, out_ch, r, stage = stage, mask=mask)

        trunk_branch = [ResidualBlock3d(out_ch, out_ch, mask=mask) for i in range(t)]
        self.trunk_branch = seq(*trunk_branch)

        out_conv = [ResidualBlock3d(out_ch, out_ch, mask=mask) for i in range(p)]
        self.out_conv = seq(*out_conv)

        self.bool_save_mask = save_mask
        self.saved_mask = None

    def forward(self, x, seq_length=None, axis=-1):
        if not self._mask:
            res = F.relu(F.max_pool3d(x, [2, 2, 1]))
            out = self.in_conv(res)
            # out += res
    
            trunk = self.trunk_branch(out)
    
            mask = self.soft_mask_branch(out)
            mask = trunk * (1 + mask)
            if self.bool_save_mask:
                self.saved_mask = mask.cpu()
    
            out = self.out_conv(mask)
        else:
            res = F.relu(F.max_pool3d(x, [2, 2, 1]))
            out = self.in_conv(res, seq_length=seq_length, axis=axis)
            # out += res
    
            trunk = self.trunk_branch(out, seq_length=seq_length, axis=axis)
    
            mask = self.soft_mask_branch(out, seq_length=seq_length, axis=axis)
            mask = trunk * (1 + mask)
            if self.bool_save_mask:
                self.saved_mask = mask.cpu()
    
            out = self.out_conv(mask, seq_length=seq_length, axis=axis)
        return out

    def get_mask(self):
        return self.saved_mask