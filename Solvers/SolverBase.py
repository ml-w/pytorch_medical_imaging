import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tqdm import *
from abc import abstractmethod

import numpy as np

class SolverBase(object):
    """

    Args:
        solver_configs (dict):
            Child class should prepare the configuration. Some keys are compulsary.

    Kwargs:
        'net_init': Initialization method. (Not implemented)


    """
    def __init__(self, solver_configs, **kwargs):
        super(SolverBase, self).__init__()

        # required
        self._optimizer         = solver_configs['optimizer']
        self._lossfunction      = solver_configs['lossfunction']
        self._net               = solver_configs['net']
        self._iscuda            = solver_configs['iscuda']

        # optional
        self._logger            = solver_configs['logger'] if 'logger' in solver_configs else None
        self._lr_decay          = solver_configs['lrdecay'] if 'lrdecay' in solver_configs else None
        self._mom_decay         = solver_configs['momdecay'] if 'momdecay' in solver_configs else None

        self._lr_decay_func     = lambda epoch: np.exp(-self._lr_decay * epoch)
        self._mom_decay_func    = lambda mom: np.max(0.2, mom * np.exp(-self._mom_decay))

        self._lr_schedular      = None
        self._called_time = 0
        self._decayed_time= 0


    def get_net(self):
        return self._net

    def get_optimizer(self):
        return self._optimizer

    def set_lr_decay(self, decay):
        self._lr_decay = decay

    def set_lr_decay_exp(self, decay):
        self._lr_decay = decay
        self._lr_schedular = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, self._lr_decay)

    def set_lr_decay_func(self, func):
        assert callable(func), "Insert function not callable!"
        self._lr_decay_func = func
        self._lr_schedular = torch.optim.lr_scheduler.LambdaLR(self._optimizer, self._lr_decay_func)

    def set_lr_decay_to_reduceOnPlateau(self, patience, factor):
        self._lr_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            factor= factor,
            patience = int(patience),
            cooldown=2,
            min_lr = 1E-6,
            threshold=0.05,
            threshold_mode='rel'
        )

    def set_momentum_decay(self, decay):
        self._mom_decay = decay

    def set_momentum_decay_func(self, func):
        assert callable(func), "Insert function not callable!"
        self._mom_dcay_func = func

    def net_to_parallel(self):
        self._net = nn.DataParallel(self._net)


    def step(self, *args):
        out = self._feed_forward(*args)
        loss = self._loss_eval(out, *args)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._called_time += 1
        return out, loss.cpu().data

    def decay_optimizer(self, *args):
        if not self._lr_schedular is None:
            self._lr_schedular.step(*args)
        if not self._mom_decay is None:
            for pg in self._optimizer.param_groups:
                pg['momentum'] = self._mom_decay_func(pg['momemtum'])
        self._decayed_time += 1
        self._log_print("Decayed optimizer...")

    def inference(self, *args):
        with torch.no_grad():
            out = self._net.forward(*list(args))
        return out


    def validation(self, val_set, gt_set, batch_size):
        validation_loss = []
        with torch.no_grad():
            dataset = TensorDataset(val_set, gt_set)
            dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False, pin_memory=False)

            for s, g in tqdm(dl, desc="Validation", position=2):
                if self._iscuda:
                        s = [ss.cuda() for ss in s] if isinstance(s, list) else s.cuda()
                        g = [gg.cuda() for gg in g] if isinstance(g, list) else g.cuda()

                if isinstance(s, list):
                    res = self._net(*s)
                else:
                    res = self._net(s)
                res = F.log_softmax(res, dim=1)
                loss = self._lossfunction(res, g.squeeze().long())
                validation_loss.append(loss.item())
        return [np.mean(np.array(validation_loss).flatten())]

    def _log_print(self, msg, level=20):
        if not self._logger is None:
            try:
                self._logger.log(level, msg)
                tqdm.write(msg)
            except:
                tqdm.write(msg)


    @staticmethod
    def _force_cuda(arg):
        return [a.cuda() for a in arg] if isinstance(arg, list) else arg.cuda()

    @abstractmethod
    def _feed_forward(self, *args):
        raise NotImplementedError

    @abstractmethod
    def _loss_eval(self, *args):
        raise NotImplementedError

