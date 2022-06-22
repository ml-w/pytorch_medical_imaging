from torch.optim import lr_scheduler

__all__ = ['DecayCAWR']

class DecayCAWR(lr_scheduler.ChainedScheduler):
    def __init__(self, optimizer, exp_factor, T0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
        assert 0 < exp_factor < 1.0, f"Exponential factor must be between 0 and 1, got {exp_factor} instead"
        lambdaLR = lr_scheduler.LambdaLR(optimizer,
                                         lambda epoch: exp_factor ** epoch,
                                         last_epoch=last_epoch,
                                         verbose=verbose)
        CAWR = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                        T_0,
                                                        T_mult,
                                                        eta_min,
                                                        last_epoch,
                                                        verbose)
        super(DecayCAWR, self).__init__([lambdaLR, CAWR])