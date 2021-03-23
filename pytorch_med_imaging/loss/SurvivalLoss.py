import torch
import torch.nn as nn

__all__ = ['CoxNLL', 'PyCoxLoss', 'WeightedCoxNLL']

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

    def forward(self,
                pred: torch.FloatTensor,
                ytime: torch.FloatTensor,
                event_status: torch.BoolTensor = None):
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
        """
        if event_status is None:
            # Assume all experienced event if not specified
            event_status = torch.ones_like(pred).bool()
        event_status = event_status.bool()

        # sort according to ytime
        _, idx = ytime.sort(0)
        sorted_ytime = ytime.gather(0, idx)
        sorted_event = event_status.view_as(ytime).gather(0, idx)
        sorted_pred = pred.gather(0, idx)
        sorted_exp_pred = torch.exp(sorted_pred)

        # flip it so cumsum sums from the back to the front
        cumsum_exp_pred = torch.flip(torch.flip(sorted_exp_pred, [1, 0]).cumsum(0), [1, 0])

        # censoring to follow up years
        censoring_vect = (sorted_ytime < self.censoring) & sorted_event

        sum_log_exp = torch.log(cumsum_exp_pred)
        log_l = sorted_pred - sum_log_exp
        log_l = log_l.mul(censoring_vect)

        N = censoring_vect.bool().float().sum()
        cox = - log_l.sum() / N
        return cox

class WeightedCoxNLL(nn.Module):
    def __init__(self, censoring: float = -1):
        r"""
        Cox harzard. Assumes no ties.

        .. math::

             $-\sum_{i}^{N} D_i \left\{ h_i - \ln \left[ \sum_{j\in R_i} \frac{T_j}{T_i} \exp(h_j) \right] \right\}$

        D_i is the censoring status of i-th individual (1 if event happened, 0 otherwise)
        R_i is the set in which they survived until event-time of i-th individual
        h_j is the network output (`pred`)
        T_i is the event time of i-th individual

        Args:
            pred (torch.Tensor):
            ytime (torch.Tensor):
            event_status (torch.Tensor):
                Censor if even did not occur. 0 if censored, 1 if not censored.
        """
        super(WeightedCoxNLL, self).__init__()
        self.censoring = censoring
        self._epx = 1E-7

    def forward(self,
                pred: torch.FloatTensor,
                ytime: torch.FloatTensor,
                event_status: torch.BoolTensor = None) -> torch.FloatTensor:
        r"""

        """
        if event_status is None:
            # Assume all experienced event if not specified
            event_status = torch.ones_like(pred).bool()
        event_status = event_status.bool()

        # sort according to ytime
        _, idx = ytime.sort(0)
        sorted_ytime = ytime.gather(0, idx)
        sorted_event = event_status.view_as(ytime).gather(0, idx)
        sorted_pred = pred.gather(0, idx)
        sorted_exp_pred = torch.exp(sorted_pred)

        # Compute weights
        sorted_invers_ytime = torch.ones_like(sorted_ytime) / sorted_ytime
        weight_mat = sorted_invers_ytime.mm(sorted_ytime.T)
        weight_mat = torch.tril(weight_mat.T, diagonal=-1).T
        sorted_exp_pred = sorted_exp_pred.expand_as(weight_mat).T

        weighted_exp_pred = weight_mat.mul(sorted_exp_pred)
        # weighted_exp_pred = sorted_exp_pred.T

        # flip it so cumsum sums from the back to the front
        sum_exp_pred = weighted_exp_pred.sum(dim=1).view_as(sorted_pred)

        # censoring to follow up years
        censoring_vect = (sorted_ytime < self.censoring) & sorted_event

        sum_log_exp = torch.log(sum_exp_pred)
        log_l = sorted_pred - sum_log_exp
        log_l = log_l.mul(censoring_vect.view_as(log_l))[:-1]  # Discard the last element bcos it should be zero

        N = censoring_vect.bool().float().sum()
        cox = - log_l.sum() / N
        return cox

class TimeDependentCoxNLL(nn.Module):
    def __init__(self, censoring: float = -1):
        r"""

        Args:
            cencering (float):
                The value which would be censored if the even time >= it.
        """
        super(TimeDependentCoxNLL, self).__init__()
        self.censoring = censoring
        self._eps = 1E-7
        self._L1_regularizer_coef = 0.2
        self._L2_regularizer_coef = 0.04


    def forward(self,
                pred: torch.FloatTensor,
                ytime: torch.FloatTensor,
                event_status: torch.BoolTensor = None):
        r"""
        Cox harzard. Assumes no ties.

        We don't want the predition to have a value that is too large, thus the L1 and L2 regularizers.

        .. math::
            $h_i(t) = h_{i(0)} + h_{i(1)}t + h_{i(2)}/t

            $\sum_{i}^{N} D_i $

        D_i is the censoring status of i-th individual (1 if event happened, 0 otherwise)
        R_i is the set in which they survived until event-time of i-th individual
        h_j is the network output (`pred`)

        Args:
            pred (torch.Tensor):
            ytime (torch.Tensor):
            event_status (torch.Tensor):
                Censor if even did not occur. 0 if censored, 1 if not censored.
        """
        if event_status is None:
            # Assume all experienced event if not specified
            event_status = torch.ones_like(pred).bool()
        event_status = event_status.bool()

        # sort according to ytime
               # sort according to ytime
        _, idx = ytime.sort(0)
        sorted_ytime = ytime.gather(0, idx)
        sorted_event = event_status.view_as(ytime).gather(0, idx)
        sorted_pred = pred.gather(0, idx)
        sorted_pred = sorted_pred[:,0] + sorted_pred[:,1] * sorted_ytime + sorted_pred[:,2] / (sorted_ytime + self._eps)
        sorted_exp_pred = torch.exp(sorted_pred)

        # flip it so cumsum sums from the back to the front
        cumsum_exp_pred = torch.flip(torch.flip(sorted_exp_pred, [1, 0]).cumsum(0), [1, 0])

        # censoring to follow up years
        censoring_vect = (sorted_ytime < self.censoring) & sorted_event

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

    def forward(self, pred, ytime, event = None):
        if event is None:
            event = torch.ones_like(pred)
        event = event.bool()

        # sort according to ytime
        _, idx = ytime.sort(0)
        sorted_ytime = ytime.gather(0, idx)
        sorted_event = event.gather(0, idx)
        if not self.censoring < 0:
            sorted_ycensor = sorted_ytime <= self.censoring
        else:
            sorted_ycensor = torch.ones_like(sorted_ytime)
        sorted_ycensor = sorted_ycensor & sorted_event
        sorted_pred = pred.gather(0, idx)

        ones = torch.ones_like(sorted_pred)
        tril = torch.tril(ones.mm(ones.T))

        sorted_pred = sorted_pred.mm(ones.T)
        sorted_exp = torch.exp(sorted_pred.mul(tril) - sorted_pred.T.mul(tril)).mul(tril).sum(dim=0)
        neg_log_exp = torch.log(sorted_exp).mul(sorted_ycensor.flatten()).sum()

        if sorted_ycensor.sum() <= 0:
            raise ArithmeticError("All input are censored.")
        return neg_log_exp / sorted_ycensor.sum()