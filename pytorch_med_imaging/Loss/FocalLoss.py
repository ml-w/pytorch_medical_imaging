import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_med_imaging.logger import Logger

class FocalLoss(nn.Module):
    r"""
    Focal loss implemented with respect to [2]_.

    .. math::

        BCE (s, g;\alpha, n) = -\alpha_n \left[ g_n \log{s_n} + (1 - g_n) \log{(1 - s_n)}\right]

        Focal (s, g;\alpha, n) =

    where:
        * :math:`s` - Input image.
        * :math:`g` - Target segmentation.
        * :math:`\alpha_l` - Weight of class :math:`l`.
        * :math:`i` - Denotes the i-th pixel.

    Args:
        epsilon (float, Optional): Smoothing factor. Default to 1e-6
        weight (float Tensor, Optional): Class weighting factor. Default to `None`.
        reeduction (str, Optional): `{'mean'|'sum'}` Method of reduction. Default to `mean`

    Returns:
        (float)


    References:
        .. [2]  Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings
                of the IEEE international conference on computer vision. 2017.

    """
    def __init__(self, with_sigmoid=True, gamma=2., reduction='mean', weight=0.8):
        super(FocalLoss, self).__init__()

        assert reduction in [None, 'mean', 'sum'], "Incorrect reduction method specified {}".format(reduction)
        assert gamma > 0, "Gamma must be greater than 0."

        self._gamma = torch.tensor(gamma)
        self._reduction = reduction
        self._weight = torch.tensor(weight)
        self._with_sigmoid = with_sigmoid
        self._logger = Logger[self.__class__.__name__]
        self._logger.debug("Set up loss with gamma: {}".format(self._gamma))
        self.register_buffer('weight', self._weight)
        self.register_buffer('gamma', self._gamma)

    def forward(self, *input):
        s, g = input
        if self._with_sigmoid:
            s = torch.sigmoid(s)

        tv = self.focal(s, g)
        if self._reduction == 'mean':
            return tv.mean()
        elif self._reduction == 'sum':
            return tv.sum()
        elif self._reduction is None:
            return tv
        else:
            self._logger.error("Incorrect setting for reduction: {}".format(self._reduction))
            raise AttributeError("Incorrect setting for reduction: {}".format(self._reduction))


    def focal(self, s, g):
        # Create index of g
        s = s.view(-1)
        g = g.view(-1)

        bce = F.binary_cross_entropy(s, g, reduction='mean')
        bce_exp = torch.exp(-bce)

        return self.weight * (1 - bce_exp) ** self.gamma * bce
