import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

import numpy as np
from logger import Logger
from tqdm import *
from abc import abstractmethod

import numpy as np

class SolverBase(object):
    def __init__(self, solver_configs):
        super(SolverBase, self).__init__()

        # required
        self._optimizer = solver_configs['optimizer']
        self._lossfunction = solver_configs['lossfunction']
        self._net = solver_configs['net']
        self._iscuda = solver_configs['iscuda']

        # optional
        self._logger = solver_configs['logger'] if 'logger' in solver_configs else None
        self._lr_decay = solver_configs['lrdecay'] if 'lrdecay' in solver_configs else None
        self._mom_decay = solver_configs['momdecay'] if 'momdecay' in solver_configs else None

        self._lr_decay_func = lambda lr: lr * np.exp(-self._lr_decay)
        self._mom_decay_func = lambda mom: np.max(0.2, mom * np.exp(-self._mom_decay))

        self._called_time = 0


    def get_net(self):
        return self._net

    def get_optimizer(self):
        return self._optimizer

    def set_lr_decay(self, decay):
        self._lr_decay = decay

    def set_lr_decay_func(self, func):
        assert callable(func), "Insert function not callable!"
        self._lr_decay_func = func

    def set_momentum_decay(self, decay):
        self._mom_decay = decay

    def set_momentum_decay_func(self, func):
        assert callable(func), "Insert function not callable!"
        self._mom_dcay_func = func


    def step(self, *args):
        out = self._feed_forward(*args)
        loss = self._loss_eval(out, *args)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # decay
        if not self._mom_decay is None or not self._lr_decay is None:
            self.decay_optimizer()

        self._called_time += 1
        return out, loss.cpu().data

    def decay_optimizer(self):
        if not self._lr_decay is None:
            for pg in self._optimizer.param_groups:
                pg['lr'] = self._lr_decay_func(pg['lr'])
        if not self._mom_decay is None:
            for pg in self._optimizer.param_groups:
                pg['momentum'] = self._mom_decay_func(pg['momemtum'])

    def inference(self, *args):
        out = self._net.forward(*list(args))
        return out


    def validation(self, val_set, gt_set, batch_size):
        with torch.no_grad():
            dataset = TensorDataset(val_set, gt_set)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False, pin_memory=False)

        validation_loss = []
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
        return np.mean(np.array(validation_loss).flatten())


    @staticmethod
    def _force_cuda(arg):
        return [a.cuda() for a in arg] if isinstance(arg, list) else arg.cuda()

    @abstractmethod
    def _feed_forward(self, *args):
        raise NotImplementedError

    @abstractmethod
    def _loss_eval(self, *args):
        raise NotImplementedError


class SegmentationSolver(SolverBase):
    def __init__(self, in_data, gt_data, net, param_optim, param_iscuda,
                 param_initWeight=None, logger=None):
        assert isinstance(logger, Logger) or logger is None, "Logger incorrect settings!"

        self._decay_init_weight = param_initWeight

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

    def decay_optimizer(self):
        super().decay_optimizer()

        sigmoid_plus = lambda x: 1. / (1. + np.exp(-x * 0.05 + 2))
        if isinstance(self._lossfunction, nn.CrossEntropyLoss):
            self._logger.log_print_tqdm('Current weight: ' + str(self._lossfunction.weight), 20)
            offsetfactor = self._called_time + self._decay_init_weight if not self._decay_init_weight is None else self._called_time
            factor =  sigmoid_plus(offsetfactor + 1) * 100
            self._lossfunction.weight.copy_(torch.as_tensor([self._r[0] * factor] + self._r[1:].tolist()))
            self._logger.log_print_tqdm('New weight: ' + str(self._lossfunction.weight), 20)

