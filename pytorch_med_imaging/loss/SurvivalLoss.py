import torch
import torch.nn as nn
from ..logger import Logger

__all__ = ['CoxNLL', 'PyCoxLoss', 'WeightedCoxBCE', 'TimeDependentCoxNLL']

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
        R_i is the set in which they survived until event-time of i-th individual, including the i-th.
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
        _, idx = ytime.sort(0, descending=True)
        sorted_ytime = ytime.gather(0, idx)
        sorted_event = event_status.view_as(ytime).gather(0, idx)
        sorted_pred = pred.gather(0, idx)

        # censoring to follow up years
        if self.censoring > 0:
            censoring_vect = (sorted_ytime < self.censoring) & sorted_event
        else:
            censoring_vect = sorted_event

        # flip it so cumsum sums from the back to the front, gamma provide numerical stability
        gamma = pred.max()
        sorted_exp_pred = torch.exp(sorted_pred - gamma)
        cumsum_exp_pred = sorted_exp_pred.cumsum(0)

        sum_log_exp = torch.log(cumsum_exp_pred) + gamma
        log_l = sorted_pred - sum_log_exp
        log_l = log_l.mul(censoring_vect)
        N = censoring_vect.bool().float().sum()
        cox = - log_l.sum() / N

        if torch.isnan(cox):
            Logger['CoxLoss'].warning("Got nan")
            Logger['CoxLoss'].error(f"{pred}\n"
                                    f"{gamma.flatten()}\n{cumsum_exp_pred.flatten()}\n{sum_log_exp.flatten()}"
                                    f"\n{log_l.flatten()}\n{N}")
        return cox

class WeightedCoxBCE(nn.Module):
    def __init__(self,
                 t_range: torch.FloatTensor or float,
                 h_range: torch.FloatTensor or float,
                 w_cox: float,
                 w_rmsdT: float,
                 w_bce: float,
                 bce_class_weight: torch.FloatTensor = None
                 ):
        r"""
        Cox loss with hazard inversely proportional

        .. math::

             $$
             \begin{equation}
               L(h;T,D) = -\frac{1}{N} \sum_i^N \left\{ \alpha \left[ h_i - \ln \sum_{j\in R(T_i)}  \exp{(h_j)}\right]
               + \beta\left| \frac{h_{\text{range}}-h_i}{h_{\text{range}}} - \frac{T_i'}{T_{\text{range}}}\right|  \right\}
               - \frac{\gamma}{N} \sum_i^N H_{\text{bce}}(h_i/h_{\text{range}},D_i)
             \end{equation}
             $$

        D_i is the censoring status of i-th individual (1 if event happened, 0 otherwise)
        R_i is the set in which they survived until event-time of i-th individual
        h_j is the network output (`pred`)
        T_i is the event time of i-th individual

        Args:
            t_range (torch.FloatTensor or float):
                Range of event time.
            h_range (torch.FloatTensor or float):
                Range of risks.
            w_cox (torch.FloatTensor or float):
                Weight of the cox loss
            w_rmsdT (torch.FloatTensor or float):
                Weight of the root-mean-square ratio loss.
            w_bce (torch.FloatTensor or float):
                Weight of the BCE loss.
            bce_class_weight (torch.FloatTensor):
                Class positive weight in the BCE loss.
        """
        super(WeightedCoxBCE, self).__init__()

        self.register_buffer('t_range', torch.as_tensor(t_range, dtype=torch.float).flatten())
        self.register_buffer('h_range', torch.as_tensor(h_range, dtype=torch.float).flatten())
        self.register_buffer('w_cox', torch.as_tensor(w_cox, dtype=torch.float).flatten())
        self.register_buffer('w_rmsdT', torch.as_tensor(w_rmsdT, dtype=torch.float).flatten())
        self.register_buffer('w_bce', torch.as_tensor(w_bce, dtype=torch.float).flatten())
        self.bce = nn.BCEWithLogitsLoss(pos_weight=bce_class_weight)
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

        #======================
        # Cox NLL
        #----------------------
        # sort according to ytime
        _, idx = ytime.sort(0, descending=True)
        sorted_ytime = ytime.gather(0, idx)
        sorted_event = event_status.view_as(ytime).gather(0, idx)
        sorted_pred = pred.gather(0, idx)

        # censoring to follow up years
        if self.t_range > 0:
            censoring_vect = (sorted_ytime < self.t_range) & sorted_event
        else:
            censoring_vect = sorted_event

        # flip it so cumsum sums from the back to the front, gamma provide numerical stability
        gamma = pred.max()
        sorted_exp_pred = torch.exp(sorted_pred - gamma)
        cumsum_exp_pred = sorted_exp_pred.cumsum(0)

        sum_log_exp = torch.log(cumsum_exp_pred) + gamma
        log_l = sorted_pred - sum_log_exp
        log_l = log_l.mul(censoring_vect)
        N = censoring_vect.bool().float().sum()
        cox = - log_l.sum() / N
        cox = self.w_cox * cox

        if torch.isnan(cox):
            Logger['CoxLoss'].warning("Got nan")
            Logger['CoxLoss'].error(f"{pred}\n"
                                    f"{gamma.flatten()}\n{cumsum_exp_pred.flatten()}\n{sum_log_exp.flatten()}"
                                    f"\n{log_l.flatten()}\n{N}")

        #======================
        # RMS ratio loss
        #----------------------
        ratio_h = (1 - sorted_pred) / self.h_range
        ratio_t = sorted_ytime / self.t_range
        ratio_t[~censoring_vect] = 1.
        diff_ratio = torch.nn.functional.l1_loss(ratio_h, ratio_t)
        rms_loss = self.w_rmsdT * diff_ratio

        #======================
        # BCE Loss
        #----------------------
        bce_l = self.w_bce * self.bce(sorted_pred, censoring_vect.float())
        return cox + rms_loss + bce_l

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
            event_status = torch.ones_like(ytime).bool()
        event_status = event_status.bool()

        # sort according to ytime
               # sort according to ytime
        _, idx = ytime.sort(0, descending=True)
        sorted_ytime = ytime.gather(0, idx)
        sorted_event = event_status.view_as(ytime).gather(0, idx)
        vec_pred = pred[:, 0] + pred[:, 1] * ytime + pred[:, 2] / (sorted_ytime + self._eps)
        sorted_pred = vec_pred.gather(0, idx)

        # censoring to follow up years
        if self.censoring > 0:
            censoring_vect = (sorted_ytime < self.censoring) & sorted_event
        else:
            censoring_vect = sorted_event

        # flip it so cumsum sums from the back to the front, gamma provide numerical stability
        gamma = pred.max()
        sorted_exp_pred = torch.exp(sorted_pred - gamma)
        cumsum_exp_pred = sorted_exp_pred.cumsum(0)

        sum_log_exp = torch.log(cumsum_exp_pred) + gamma
        log_l = sorted_pred - sum_log_exp
        log_l = log_l.mul(censoring_vect)
        N = censoring_vect.bool().float().sum()
        cox = - log_l.sum() / N
        if torch.isnan(cox):
            Logger['CoxLoss'].warning("Got nan")
            Logger['CoxLoss'].error(f"{pred}\n{vec_pred.flatten()}\n"
                                    f"{gamma.flatten()}\n{cumsum_exp_pred.flatten()}\n{sum_log_exp.flatten()}"
                                    f"\n{log_l.flatten()}\n{N}")

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