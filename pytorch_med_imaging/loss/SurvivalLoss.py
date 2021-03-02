import torch
import torch.nn as nn

__all__ = ['CoxNLL', 'PyCoxLoss']

class CoxNLL(nn.Module):
    def __init__(self, censoring: float = -1):
        r"""

        Args:
            cencering (float):
                The value which would be censored if the even time >= it.
        """
        super(CoxNLL, self).__init__()
        self.censoring = censoring
        self._eps = 1E-7
        self._L1_regularizer_coef = 0.2
        self._L2_regularizer_coef = 0.04



    def _L1(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._L1_regularizer_coef * tensor.sum()

    def _L2(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._L2_regularizer_coef * tensor.square().sum()

    def forward(self, pred, ytime, event_status=None):
        r"""
        Cox harzard. Assumes no ties.

        We don't want the predition to have a value that is too large, thus the L1 and L2 regularizers.

        .. math::

             $\sum_{i}^{N} D_i \left\{ h_i - \ln \left[ \sum_{j\in R_i} \exp(h_j) \right] \right\}$

        D_i is the censoring status of i-th individual (1 if event happened, 0 otherwise)
        R_i is the set in which they survived until event-time of i-th individual
        h_j is the network output (`pred`)

        Args:
            pred (torch.Tensor):
            ytime (torch.Tensor):
            event_status (torch.Tensor):
                Censor if even did not occur. 0 if censored, 1 if not censored.

        Returns:

        """
        if event_status is None:
            event_status = torch.zeros_like(pred).bool()
        event_status = event_status.bool()

        # sort according to ytime
        _, idx = ytime.sort(0)
        # Logger['CoxNLL'].error(f"{idx}")
        # Logger['CoxNLL'].error(f"{ytime}")
        # Logger['CoxNLL'].error(f"{event_status}")
        sorted_ytime = ytime.gather(0, idx)
        sorted_ycensor = event_status.view_as(ytime).gather(0, idx)
        sorted_pred = pred.gather(0, idx)
        sorted_exp_pred = torch.exp(sorted_pred)

        # flip it so cumsum sums from the back to the front
        cumsum_exp_pred = torch.flip(torch.flip(sorted_exp_pred, [1, 0]).cumsum(0), [1, 0])

        # censoring to follow up years
        censoring_vect = (sorted_ytime < self.censoring) | (sorted_ycensor)

        sum_log_exp = torch.log(cumsum_exp_pred)
        log_l = sorted_pred - sum_log_exp
        log_l = log_l.mul(censoring_vect)

        N = censoring_vect.bool().float().sum()
        cox = - log_l.sum() / N
        return cox

class PyCoxLoss(nn.Module):
    def __init__(self, censoring: float = -1):
        super(PyCoxLoss, self).__init__()
        self.censoring = censoring
        self._eps = 1E-7

    def forward(self, pred, ytime):

        # sort according to ytime
        _, idx = ytime.sort(0)
        sorted_ytime = ytime.gather(0, idx)
        if not self.censoring < 0:
            sorted_ycensor = sorted_ytime <= self.censoring
        else:
            sorted_ycensor = torch.ones_like(sorted_ytime)
        sorted_pred = pred.gather(0, idx)

        ones = torch.ones_like(sorted_pred)
        tril = torch.tril(ones.mm(ones.T))

        sorted_pred = sorted_pred.mm(ones.T)
        sorted_exp = torch.exp(sorted_pred.mul(tril) - sorted_pred.T.mul(tril)).mul(tril).sum(dim=0)
        neg_log_exp = torch.log(sorted_exp).mul(sorted_ycensor.flatten()).sum()

        if sorted_ycensor.sum() <= 0:
            raise ArithmeticError("All input are censored.")
        return neg_log_exp / sorted_ycensor.sum()