import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

from .logger import Logger
from tqdm import *
import numpy as np
from abc import abstractmethod

class SolverBase(object):
    def __init__(self, solver_configs):
        super(SolverBase, self).__init__()

        # required
        self._optimizer = solver_configs['optimizer']
        self._lossfunction = solver_configs['lossfunction']
        self._net = solver_configs['net']
        self._iscuda = solver_configs['iscuda']

        # optional
        self._logger = solver_configs['logger'] if solver_configs.has_key('logger') else None


    def step(self, *args):
        out = self._feed_forward(*args)
        loss = self._loss_eval(out, *args)
        self._optimizer.zero_grad()
        self._lossfunction.backward()
        self._optimizer.step()
        return out, loss.cpu().data


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
    def __init__(self, gt_data, net, param_optim, param_iscuda,
                 param_initWeight=None, logger=None):
        assert isinstance(logger, Logger) or logger is None, "Logger incorrect settings!"
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
        del valcountpair # free RAM

        # calculate init-factor
        if not param_initWeight is None:
            factor =  sigmoid_plus(param_initWeight + 1) * 100
        else:
            factor = 1
        weights = torch.as_tensor([r[0] * factor] + r[1:].tolist())
        logger.log_print_tqdm("Initial weight factor: " + str(weights))

        lossfunction = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.SGD(param_optim)
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
