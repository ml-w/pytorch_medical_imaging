"""
Implemented according to:
    Oktay, Ozan, et al. "Attention U-Net: Learning Where to Look for the Pancreas."
    arXiv preprint arXiv:1804.03999 (2018).

For 2D attention gates only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, inchan, gating_chan, subsample=2):
        super(AttentionBlock, self).__init__()

        interchan = inchan // 2 if inchan / 2 > 0 else 1
        self.W = nn.Sequential(
            nn.Conv2d(inchan, inchan, kernel_size=1),
            nn.BatchNorm2d(inchan)
        )

        self.theta = nn.Conv2d(inchan, interchan, kernel_size=subsample, stride=subsample, padding=0, bias=False)
        self.phi = nn.Conv2d(gating_chan ,interchan, kernel_size=1, stride=1, bias=True)
        self.psi = nn.Conv2d(interchan, 1, kernel_size=1, padding=0, bias=True)

        # TODO: Should have kamming initialization here for children of self

    def _concatentation(self, x, g):
        x_size = x.shape
        assert x.shape[0] == g.shape[0], "Unequal batch size."

        theta_x = self.theta(x)
        phi_g = F.upsample(self.phi(g), size=theta_x.shape[2:], mode='bilinear')
        f = F.relu(theta_x + phi_g, inplace=True)
        f = F.sigmoid(self.psi(f))

        f = F.upsample(f, size=x_size[2:], mode='bilinear')
        y = f.expand_as(x) * x
        W_y = self.W(y)
        return W_y, f


    def forward(self, *input):
        return self._concatentation(*input)

