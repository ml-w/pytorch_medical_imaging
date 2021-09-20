from ..UNet_p import UNet_p
from typing import Union, Optional, Iterable

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['UNetFCAttention_p']

class UNetFCAttention_p(UNet_p):
    r"""
    This is the version of UNet with fully connected attention.

    Example:
        1. UNetLocTexHist_p(1, 2, layers=5)
        --------------------------------------------------------------------------------
                 Layer (type)          Output Shape         Param #     Tr. Param #
        ================================================================================
           ReflectiveDoubleConv-1     [2, 64, 128, 128]          37,824          37,824
                         Linear-2              [2, 300]          61,500          61,500
                      LayerNorm-3              [2, 300]             600             600
                           ReLU-4              [2, 300]               0               0
                         Linear-5              [2, 600]         180,600         180,600
                      LayerNorm-6              [2, 600]           1,200           1,200
                           ReLU-7              [2, 600]               0               0
                           Down-8      [2, 128, 64, 64]         221,952         221,952
                           Down-9      [2, 256, 32, 32]         886,272         886,272
                          Down-10      [2, 512, 16, 16]       3,542,016       3,542,016
                          Down-11       [2, 1024, 8, 8]      14,161,920      14,161,920
                          Down-12       [2, 2048, 4, 4]      56,635,392      56,635,392
                        Linear-13              [2, 128]          76,928          76,928
                        Linear-14              [2, 256]         153,856         153,856
                        Linear-15              [2, 512]         307,712         307,712
                        Linear-16             [2, 1024]         615,424         615,424
                            Up-17       [2, 1024, 8, 8]      60,830,720      60,830,720
                            Up-18      [2, 512, 16, 16]      15,211,008      15,211,008
                            Up-19      [2, 256, 32, 32]       3,804,416       3,804,416
                            Up-20      [2, 128, 64, 64]         951,936         951,936
                            Up-21     [2, 64, 128, 128]         238,400         238,400
                        Conv2d-22      [2, 2, 128, 128]             130             130
        ================================================================================
        Total params: 157,919,806
        Trainable params: 157,919,806
        Non-trainable params: 0
        --------------------------------------------------------------------------------

    """
    def __init__(self,
                 in_chan: int,
                 out_chan: int,
                 fc_inchan: int,
                 layers: Optional[int] = 4,
                 down_mode: Optional[str] = 'avg',
                 up_mode: Optional[str] = 'learn',
                 dropout: float = 0.1):
        super(UNetFCAttention_p, self).__init__(in_chan, out_chan, layers=layers,
                                                down_mode=down_mode, up_mode=up_mode, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(fc_inchan, 300),
            nn.LayerNorm(300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 600),
            nn.LayerNorm(600),
            nn.ReLU(inplace=True)
        )

        down_chs = [self._start_chans * 2 ** (i + 1) for i in range(layers - 1)]
        self.fcs = nn.ModuleList([nn.Linear(600, dc) for dc in down_chs])

    def forward(self, x, fv):
        r"""
        Rewritten the forward function to acomodate the textural-positional vector `fv`.
        """
        x = self.inconv(x)

        fv = self.fc(fv)
        short_conn = [x]
        for i, d in enumerate(self.downs):
            x = d(x)
            # Except last layer
            if i < self._layers - 1: # 0, 1, 2
                short_conn.append(x)

        X = [short_conn[0]]
        for _x, _fc in zip(short_conn[1:], self.fcs):
            _fv = _fc(fv).expand_as(_x.permute(2, 3, 0, 1)).permute(2, 3, 0, 1)
            _x = _x * F.relu(_fv, True)
            X.append(_x)

        for i, u in enumerate(self.ups):
            x = u(X[- i - 1], x)
        x = self.lastconv(x)