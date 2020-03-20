import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    r"""
    This function calculates the multi-class soft dice of the input. Implementation follows the reference [1]_.

    .. math::

        D_{\text{soft}} (s, g) = \frac{2\sum_l \alpha_l \sum_i s^i_l g^i_l}
                                        {\sum_l \alpha_l \sum_i (s^i_l + g^i_l)}

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
        .. [1]  Sudre, Carole H., et al. "Generalised dice overlap as a deep learning loss function for highly
                unbalanced segmentations." Deep learning in medical image analysis and multimodal learning for clinical
                decision support. Springer, Cham, 2017. 240-248.

    """
    def __init__(self, epsilon=1e-6, weight=None, reduction='mean'):
        super(SoftDiceLoss, self).__init__()
        self.reduction=reduction
        self.epsilon = epsilon
        self.register_buffer('weight', weight)

    def _soft_dice(self, input, target):
        """
        Actual implementation of soft dice.
        """
        n = input.size(0)
        c = input.size(1)

        dim = target.dim()
        if input.dim() != dim + 1:
            raise ValueError('Expected target dimension {} but get {}.'.format(dim+1, target.dim()))

        ret = []
        for j in range(n):
            a = []
            b = []
            for i in range(c):
                s = input[j,i]
                g = (target[j] == i).float()
                if not self.weight is None:
                    w = self.weight[i]
                else:
                    w = 1.

                a.append((s * g).sum() * w)
                b.append((s + g).sum() * w)
            a = torch.stack(a)
            b = torch.stack(b)
            ret.append((2 * torch.sum(a) +  self.epsilon) / (torch.sum(b) + self.epsilon))

        ret = torch.stack(ret)
        ret = torch.ones_like(ret) - ret
        if self.reduction == 'mean':
            ret = torch.mean(ret)
        elif self.reduction == 'sum':
            ret = torch.sum(ret)
        else:
            raise ValueError("Wrong reduction type {}.".format(self.reduction))
        return ret

    def forward(self, input, target):
        r"""
        Forward for DICE loss.

        Args:
            input (float Tensor): Input size `(B x C x H x W)` or `(B x C x D x H x W)`.
            target (long Tensor): Target should have one less dimension as `input`.

        Returns:
            (torch.Tensor)
        """
        dim = input.dim()
        if dim < 2:
            raise ValueError('Expected 2 or more dimensions (got {})'.format(dim))

        if input.size(0) != target.size(0):
            raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                             .format(input.size(0), target.size(0)))

        n = input.size(0)
        c = input.size(1)
        input = input.contiguous().view(n, c, -1)
        target = target.contiguous().view(n, -1)
        ret = self._soft_dice(input, target)
        return ret