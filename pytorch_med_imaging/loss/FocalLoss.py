import torch
import torch.nn as nn
import torch.nn.functional as F
from mnts.mnts_logger import MNTSLogger

class BinaryFocalLoss(nn.Module):
    r"""
    Focal loss implemented with respect to [2]_.

    .. math::

        BCE (s, g;\alpha, n) = -\alpha_n \left[ g_n \log{s_n} + (1 - g_n) \log{(1 - s_n)}\right]

        Focal (s, g;\alpha, \gamma, n) = (1 - s)^\gamma \cdot BCE(s, g;\alpha, n)

    where:
        * :math:`s` - Input prediction `x`.
        * :math:`g` - Target segmentation `target`.
        * :math:`\alpha_l` - Weight of class :math:`l`.
        * :math:`i` - Denotes the i-th pixel.

    Args:
        with_sigmoid (bool): Whether to use sigmoid before calculating focal loss. Default to `True`.
        epsilon (float, Optional): Smoothing factor. Default to 1e-6
        alpha (float Tensor, Optional): Class weighting factor. Default to `None`.
        reduction (str, Optional): `{'mean'|'sum'}` Method of reduction. Default to `mean`

    Returns:
        (float)


    References:
        .. [2]  Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings
                of the IEEE international conference on computer vision. 2017.

    """
    def __init__(self, with_sigmoid=True, gamma=2., reduction='mean', alpha=0.8):
        super(BinaryFocalLoss, self).__init__()

        assert reduction in [None, 'none', 'mean', 'sum'], "Incorrect reduction method specified {}".format(reduction)
        assert gamma > 0, "Gamma must be greater than 0."

        self._gamma = torch.tensor(gamma)
        self._reduction = reduction
        self._with_sigmoid = with_sigmoid
        self._weight = torch.tensor(alpha)
        self.register_buffer('weight', self._weight)
        self.register_buffer('gamma', self._gamma)

    def forward(self, x: torch.Tensor, targets: torch.Tensor):
        if self._with_sigmoid:
            x = torch.sigmoid(x)

        tv = self.focal(x, targets)
        if self._reduction == 'mean':
            return tv.mean()
        elif self._reduction == 'sum':
            return tv.sum()
        elif self._reduction is None or self._reduction == 'none':
            return tv
        else:
            raise AttributeError("Incorrect setting for reduction: {}".format(self._reduction))


    def focal(self, x: torch.Tensor, targets: torch.Tensor):
        # Create index of g
        ce_loss = F.binary_cross_entropy(x, targets, reduction="none")
        p_t = x * targets + (1 - x) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.weight >= 0:
            alpha_t = self.weight * targets + (1 - self.weight) * (1 - targets)
            loss = alpha_t * loss

        return loss
