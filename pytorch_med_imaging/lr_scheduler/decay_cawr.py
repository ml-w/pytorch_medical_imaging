from torch.optim import lr_scheduler
from bisect import bisect_right

__all__ = ['DecayCAWR', 'DecayCAWR_n_EXP']

class DecayCAWR(lr_scheduler.CosineAnnealingWarmRestarts):
    r"""This lr_scheduler is a Cosine Annealing Warm Restarts with exponential decay.

    Args:
        optimizer (torch.optim.Optimizer):
            Optimizer.
        exp_factor (float):
            The decay multiplicative factor between 0 to 1.
        T_0 (int):
            Number of iterations for the first restart.
        T_mult (int, optional):
            A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional):
            Minimum learning rate. Default: 0.
        last_epoch (int, optional):
            The index of last epoch. Default: -1.
        verbose (bool):
            If ``True``, prints a message to stdout for each update. Default: ``False``.
    """
    def __init__(self, optimizer, exp_factor, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
        if not (0 < exp_factor < 1.0):
            raise ValueError("Expected exp_factor to be between 0 - 1, but got {}.".format(exp_factor))
        self.exp_factor = exp_factor
        super(DecayCAWR, self).__init__(optimizer, T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch,
                                        verbose=verbose)

    def get_lr(self):
        base_lr = super(DecayCAWR, self).get_lr()
        return [blr * self.exp_factor ** self.last_epoch for blr in base_lr]


class DecayCAWR_n_EXP(lr_scheduler.SequentialLR):
    r"""This lr_scheduler sequentially calls DecayCAWR and then exponential

    Args:
        optimizer (torch.optim.Optimizer):
            Optimizer.
        exp_factor_dcawr (float):
            The decay multiplicative factor between 0 to 1 for DecayCAWR.
        gamma (faoat):
            The multiplcative factor for ExponentialLR.
        T_0 (int):
            Number of iterations for the first restart.
        T_cut (int);
            The number of period before invoking the Exponential LR.
        T_mult (int, optional):
            A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional):
            Minimum learning rate. Default: 0.
        last_epoch (int, optional):
            The index of last epoch. Default: -1.
        verbose (bool):
            If ``True``, prints a message to stdout for each update. Default: ``False``.

    """
    def __init__(self, optimizer, exp_factor_dcawr, gamma, T_0, T_cut, **kwargs):
        if T_cut < 1:
            msg = f"Expect T_cut to be an interger > 1, but got {T_cut}"
            raise AttributeError(msg)
        T_mult = kwargs.get('T_mult', 1)
        dcawr = DecayCAWR(optimizer, exp_factor_dcawr, T_0, **kwargs)
        exp = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        cut_point = np.cumsum(T_0 * T_mult ** np.arange(T_cut))[-1] + 1
        super(DecayCAWR_n_EXP, self).__init__(optimizer, [dcawr, exp], [cut_point], last_epoch=-1, verbose=kwargs.get('verbose', False))

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            self._last_lr = self._schedulers[idx - 1].get_last_lr()
            self._schedulers[idx].step()
        else:
            self._schedulers[idx].step()
            self._last_lr = self._schedulers[idx].get_last_lr()
