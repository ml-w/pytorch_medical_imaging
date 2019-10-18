from .SolverBase import SolverBase
from logger import Logger

from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import *

class ClassificationSolver(SolverBase):
    def __init__(self,in_data, gt_data, net, param_optim, param_iscuda,
                 param_initWeight=None, logger=None):
        assert isinstance(logger, Logger) or logger is None, "Logger incorrect settings!"


        self._decay_init_weight = param_initWeight

        solver_configs = {}
        # check unique class in gt
        logger.log_print_tqdm("Detecting number of classes...")
        numOfClasses = len(gt_data.get_unique_values())
        numOfClasses = 2 if numOfClasses < 2 else numOfClasses
        numOfClasses = 2 #TODO: Temp fix
        logger.log_print_tqdm("Find %i classes.."%(numOfClasses))


        inchan = in_data[0].size()[0]

        net = net(inchan, numOfClasses)

        # Create optimizer and loss function
        lossfunction = nn.CrossEntropyLoss()
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

        super(ClassificationSolver, self).__init__(solver_configs)


    def _feed_forward(self, *args):
        s, g = args

        if self._iscuda:
            s = self._force_cuda(s)

        out = self._net.forward(s)
        return out

    def _loss_eval(self, *args):
        out, s, g = args
        if self._iscuda:
            g = self._force_cuda(g)

        loss = self._lossfunction(out.squeeze(), g.squeeze().long())
        return loss

    def validation(self, val_set, gt_set, batch_size):
        with torch.no_grad():
            dataset = TensorDataset(val_set, gt_set)
            dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False, pin_memory=False)

            decisions = []
            validation_loss = []
            for s, g in tqdm(dl, desc="Validation", position=2):
                if self._iscuda:
                        s = [ss.cuda() for ss in s] if isinstance(s, list) else s.cuda()
                        g = [gg.cuda() for gg in g] if isinstance(g, list) else g.cuda()

                if isinstance(s, list):
                    res = self._net(*s)
                else:
                    res = self._net(s)
                # res = torch.(res, dim=1)
                while res.dim() < 2:
                    res = res.unsqueeze(0)
                dic = torch.argmax(torch.softmax(res, dim=1), dim=1)
                decisions.extend([guess == truth for guess, truth in zip(dic.tolist(), g.tolist())])
                loss = self._lossfunction(res, g.long())
                validation_loss.append(loss.item())

            # Compute accuracies
            acc = float(decisions.count(True)) / float(len(decisions))
            validation_loss = np.mean(np.array(validation_loss).flatten())
            self._logger.log_print_tqdm("Validation Result - ACC: %.05f, VAL: %.05f"%(acc, validation_loss))
            return validation_loss, acc

