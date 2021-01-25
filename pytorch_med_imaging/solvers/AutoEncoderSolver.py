import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from .SolverBase import SolverBase
from pytorch_med_imaging.logger import Logger

__all__ = ['AutoEncoderSolver']


class AutoEncoderSolver(SolverBase):
    def __init__(self, in_data, gt_data, net, param_optim, param_iscuda,
                 param_initWeight=None, logger=None, **kwargs):
        assert isinstance(logger, Logger) or logger is None, "Logger incorrect settings!"

        if logger is None:
            logger = Logger[self.__class__.__name__]

        self._decay_init_weight = param_initWeight if not param_initWeight is None else 0

        solver_configs = {}
        # Create network
        if not hasattr(net, 'forward'):
            try:
                if isinstance(in_data[0], tuple) or isinstance(in_data[0], list):
                    inchan = in_data[0][0].shape[0]
                else:
                    inchan = in_data[0].size()[0]
            except AttributeError:
                # retreat to 1 channel
                logger.warning("Retreating to 1 channel.")
                inchan = 1
            except Exception as e:
                logger.log_traceback(e)
                logger.warning("Error encountered when calculating number of channels.")
                raise ArithmeticError("Error encountered when calculating number of channels.")
            net = net(inchan, inchan)

        # Create optimizer and loss function
        lossfunction = nn.SmoothL1Loss() #TODO: Allow custom loss function
        optimizer = optim.Adam(net.parameters(), lr=param_optim['lr'])

        iscuda = param_iscuda
        if param_iscuda:
            lossfunction = lossfunction.cuda()
            net = net.cuda()

        solver_configs['optimizer'] = optimizer
        solver_configs['lossfunction'] = lossfunction
        solver_configs['net'] = net
        solver_configs['iscuda'] = iscuda
        solver_configs['logger'] = logger

        super(AutoEncoderSolver, self).__init__(solver_configs, **kwargs)


    def _feed_forward(self, *args):
        s, g = args

        # Handle list elements
        if (isinstance(s, list) or isinstance(s, tuple)) and len(s) > 1:
            s = [Variable(ss).float() for ss in s]
        else:
            s = Variable(s).float()

        if self._iscuda:
            s = self._force_cuda(s)

        if isinstance(s, list):
            out = self._net.forward(*s)
        else:
            out = self._net.forward(s)

        return out

    def _loss_eval(self, *args):
        out, s, g = args
        if (isinstance(g, list) or isinstance(g, tuple)) and len(g) > 1:
            g = [Variable(gg, requires_grad=False) for gg in g]
        else:
            g = Variable(g, requires_grad=False)

        if self._iscuda:
            g = self._force_cuda(g)

        loss = self._lossfunction(out, g)
        return loss


