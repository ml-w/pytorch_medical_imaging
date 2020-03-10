import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from .SolverBase import SolverBase
from logger import Logger


class SegmentationSolver(SolverBase):
    def __init__(self, in_data, gt_data, net, param_optim, param_iscuda,
                 param_initWeight=None, logger=None):
        assert isinstance(logger, Logger) or logger is None, "Logger incorrect settings!"

        self._decay_init_weight = param_initWeight if not param_initWeight is None else 0

        solver_configs = {}
        # check unique class in gt
        logger.log_print_tqdm("Detecting number of classes...")
        valcountpair = gt_data.get_unique_values_n_counts()
        classes = list(valcountpair.keys())
        numOfClasses = len(classes)
        logger.log_print_tqdm("Find %i classes: %s"%(numOfClasses, classes))

        # calculate empty label ratio for updating loss function weight
        r = []
        sigmoid_plus = lambda x: 1. / (1. + np.exp(-x * 0.05 + 2))
        for c in classes:
            factor = float(np.prod(np.array(gt_data.size())))/float(valcountpair[c])
            r.append(factor)
        r = np.array(r)
        r = r / r.max()
        self._r = r
        del valcountpair # free RAM

        # calculate init-factor
        if not param_initWeight is None:
            factor =  sigmoid_plus(param_initWeight + 1) * 100
        else:
            factor = 1
        weights = torch.as_tensor([r[0] * factor] + r[1:].tolist())
        self.loss_init_weights = weights.cpu()
        logger.log_print_tqdm("Initial weight factor: " + str(weights))

        # Create network
        try:
            if isinstance(in_data[0], tuple) or isinstance(in_data[0], list):
                inchan = in_data[0][0].shape[0]
            else:
                inchan = in_data[0].size()[0]
        except AttributeError:
            # retreat to 1 channel
            logger.log_print_tqdm("Retreating to 1 channel.", 30)
            inchan = 1
        except Exception as e:
            logger.log_print_tqdm(str(e), 40)
            logger.log_print_tqdm("Terminating", 40)
            raise ArithmeticError("Cannot compute in channel!")

        net = net(inchan, numOfClasses)

        # Create optimizer and loss function
        lossfunction = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.SGD(net.parameters(), lr=param_optim['lr'], momentum=param_optim['momentum'])
        iscuda = param_iscuda
        if param_iscuda:
            lossfunction = lossfunction.cuda()
            net = net.cuda()

        solver_configs['optimizer'] = optimizer
        solver_configs['lossfunction'] = lossfunction
        solver_configs['net'] = net
        solver_configs['iscuda'] = iscuda
        solver_configs['logger'] = logger

        super(SegmentationSolver, self).__init__(solver_configs)


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

        loss = self._lossfunction(out, g.squeeze().long())
        return loss

    def decay_optimizer(self, *args):
        """
        Function is called every epoch, the weight of the optimizer is fine tuned such that initially the loss
        weights are balanced by the count of each class, the weight will quickly shift towards favoring empty pixel
        for the first dozens of epoch until the all weights slowly converge to 1.

        The weight follows the equation:

        .. math::

            s(x;x_0, s, d) = x_0 + \frac{(1 - x_0)}{(1 + e^{-(x - 2d) / s}}

        Where:
        * :math:`x` - Input weight.
        * :math:`x_0` - Initial count, used when training is not starting from 0th epoch, say for after loading cp.
        * :math:`s` - Stretch, controls the slope of the convergence
        * :math:`d` - Delay, controls which epochs starts the convergence

        Args:
            *args:

        Returns:

        Todo:
            * Allow loading parameters.

        .. note::
            Currently the coefficient :math:`s` and :math:`d` are set to 10 and 35 (epochs) respectively.

        """
        super().decay_optimizer(*args)

        cap = 0.5
        sigmoid_plus = lambda x, init, stretch, delay: (init + (1 - init) * 1. / \
                                                       (1 + np.exp(- x / stretch + delay * 2 / stretch))) * cap
        if isinstance(self._lossfunction, nn.CrossEntropyLoss):
            self._logger.log_print_tqdm('Current weight: ' + str(self._lossfunction.weight), 20)
            offset = self._decayed_time + self._decay_init_weight
            new_weight = torch.as_tensor([sigmoid_plus(offset, self._r[i], 10, 35) for i in range(len(self._r))])
            self._lossfunction.weight.copy_(new_weight)
            self._logger.log_print_tqdm('New weight: ' + str(self._lossfunction.weight), 20)

