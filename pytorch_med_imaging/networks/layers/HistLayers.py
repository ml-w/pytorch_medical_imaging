import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Gaussian(nn.Module):
    r"""
    Compaute a Gaussian:

    .. math::

        $$ g(x;\mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}}\exp \left[\frac{-(x - \mu)^2}{\sigma^2} \right] $$
    """
    def __int__(self, mu: torch.FloatTensor, sigma: torch.FloatTensor):
        super(Gaussian, self).__int__()

        if not torch.is_tensor(mu):
            mu = torch.as_tensor([mu])
        if not torch.is_tensor(sigma):
            sigma = torch.as_tensor([sigma])

        self.register_buffer('mu', mu)
        self.register_buffer('sigma', sigma)
        self.register_buffer('pi', torch.as_tensor([np.pi]))


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x.sub(self.mu).pow(2).mul(-1).mul(self.sigma.pow(-2.)).exp(). \
                mul(self.sigma.mul(self.pi.mul(2).pow(.5)).pow(-1.))


class HistLayer(nn.Module):
    r"""
    Computes the histogram of a 2D input tensor

    References:
        [1] Yusuf, Ibrahim, George Igwegbe, and Oluwafemi Azeez. "Differentiable Histogram
            with Hard-Binning." arXiv preprint arXiv:2012.06311 (2020).
    """
    def __init__(self, num_bins, min_val, max_val, sigma, d=2):
        super(HistLayer, self).__init__()

        self.register_buffer('num_bins', num_bins)
        self.register_buffer('min_val', min_val)
        self.register_buffer('max_val', max_val)
        self.register_buffer('sigma', sigma)

        x = torch.linspace(self.min_val, self.max_val, self.num_bins)
        self.bins = nn.ModuleDict({xx: nn.Conv2d() for xx in x})

    def forward(self, x:torch.FloatTensor) -> torch.FloatTensor:
        pass

