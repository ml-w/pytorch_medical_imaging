import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SpectralLoss']


class SpectralLoss(nn.Module):
    r"""Log-magnitude spectral L1 loss for frequency-domain alignment.

    Computes the L1 distance between the log-scaled 2D FFT magnitude spectra of the
    prediction and the target. Penalising errors in the log-magnitude spectrum encourages
    the generator to reproduce high-frequency content (edges, bone boundaries) that standard
    pixel-level losses under-weight.

    .. math::

        \mathcal{L}_{\text{freq}}(y, \hat{y}) =
            \| \log(|\mathcal{F}(\hat{y})| + \varepsilon) -
               \log(|\mathcal{F}(y)|        + \varepsilon) \|_1

    where :math:`\mathcal{F}` denotes the 2D real FFT and :math:`\varepsilon` is a small
    constant for numerical stability.

    Args:
        eps (float, Optional):
            Stability constant added before taking the log. Default to ``1e-8``.

    Examples:
        >>> loss_fn = SpectralLoss()
        >>> pred   = torch.randn(2, 1, 64, 64)
        >>> target = torch.randn(2, 1, 64, 64)
        >>> loss   = loss_fn(pred, target)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            pred (torch.Tensor):
                Predicted image tensor. Shape :math:`(B \times C \times H \times W)`.
            target (torch.Tensor):
                Ground-truth image tensor. Same shape as ``pred``.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        pred_mag   = torch.abs(torch.fft.rfft2(pred,   norm='ortho'))
        target_mag = torch.abs(torch.fft.rfft2(target, norm='ortho'))
        return F.l1_loss(torch.log(pred_mag   + self.eps),
                         torch.log(target_mag + self.eps))
