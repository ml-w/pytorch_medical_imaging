import torch
import torch.nn as nn

__all__ = ['CoxNLL']

class CoxNLL(nn.Module):
    def __init__(self, censoring: float = -1):
        r"""

        Args:
            cencering (float):
                The value which would be censored if the even time >= it.
        """
        super(CoxNLL, self).__init__()
        self.censoring = censoring
        self._eps = 1E-14

    def forward(self, pred, ytime, ycensor=None):
        r"""
        Cox harzard. Assumes no additional censoring needed.

        .. math::

             $\sum_{i}^{N} D_i \left\{ h_i - \ln \left[ \sum_{j\in R_i} \exp(h_j) \right] \right\}$

        D_i is the censoring status of i-th individual
        R_i is the set in which they survived until event-time of i-th individual
        h_j is the network output (`pred`)

        Args:
            pred (torch.Tensor):
            ytime (torch.Tensor):
            ycensor (torch.Tensor):

        Returns:

        """
        if ycensor is None:
            ycensor = torch.ones_like(pred)

        # sort according to ytime
        _, idx = ytime.sort(0)
        sorted_ytime = ytime.gather(0, idx)
        sorted_ycensor = ycensor.gather(0, idx)
        sorted_pred = pred.gather(0, idx)
        sorted_exp_pred = torch.exp(sorted_pred)

        cumsum_exp_pred = torch.flip(torch.flip(sorted_exp_pred, [1, 0]).cumsum(0), [1, 0])
        censoring_vect = (sorted_ytime < self.censoring).mul(sorted_ycensor)

        sum_log_exp = torch.log(cumsum_exp_pred)
        log_l = sorted_pred - sum_log_exp
        log_l = log_l.mul(censoring_vect)

        N = censoring_vect.bool().float().sum()
        return - log_l.sum() / N

    @staticmethod
    def _indicator_mat(x):
        """
        Create tril matrix for censoring.
        """
        n_sample = x.size(0)
        one_hot = torch.ones(n_sample, n_sample)
        indicator_mat = torch.tril(one_hot)

        return indicator_mat

