from torch.optim import lr_scheduler

__all__ = ['DecayCAWR']

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
