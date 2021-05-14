from .SolverBase import SolverBase
from pytorch_med_imaging.logger import Logger


from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np
from tqdm import *

import pandas as pd

__all__ = ['ClassificationSolver']

class ClassificationSolver(SolverBase):
    def __init__(self,in_data, gt_data, net, param_optim, param_iscuda,
                 param_initWeight=None, logger=None, **kwargs):
        """
        Solver for classification tasks.

        Args:
            in_data (PMIDataBase):
                Tensor of input data.
            gt_data (PMIDataBase):
                Tensor of output data.
            net (class):
                Network modules.
            param_optim (dict):
                Dictionary of the optimizer parameters. Should include key 'lr'.
            param_iscuda (bool):
                Settings to use CUDA or not.
            param_initWeight (int, Optional):
                Initial weight for loss function.
            logger (Logger, Optional):
                Logger. If no logger provide, log will be output to './temp.log'
            **kwargs:
                Additional dictionary item pass to base class.

        Kwargs:
            For details to kwargs, see :class:`SolverBase`.

        Returns:
            :class:`ClassificaitonSolver` object
        """
        assert isinstance(logger, Logger) or logger is None, "Logger incorrect settings!"

        if logger is None:
            logger = Logger[self.__class__.__name__]

        self._decay_init_weight = param_initWeight

        solver_configs = {}
        # check unique class in gt
        logger.info("Detecting number of classes...")
        numOfClasses = len(gt_data.get_unique_values())
        numOfClasses = 2 if numOfClasses < 2 else numOfClasses
        logger.info("Find %i classes.."%(numOfClasses))



        if not hasattr(net, 'forward'):
            self._logger.info("Creating network object...")
            inchan = in_data[0].size()[0]
            net = net(inchan, numOfClasses)

        # Create optimizer and loss function
        lossfunction = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(net.parameters(), lr=param_optim['lr'], momentum=param_optim['momentum'])
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

        super(ClassificationSolver, self).__init__(solver_configs, **kwargs)


    def _feed_forward(self, *args):
        s, g = args
        try:
            s = self._match_type_with_network(s)
        except:
            self._logger.exception("Failed to match input to network type. Falling back.")
            if self.iscuda:
                s = self._force_cuda(s)
                self._logger.debug("_force_cuda() typed data as: {}".format(
                    [ss.dtype for ss in s] if isinstance(s, list) else s.dtype))


        # if isinstance(s, list):
        #      [ss.requires_grad_() for ss in s]
        # else:
        #     s.requires_grad_()
        # Variable is deprecated in pyTorch v1.5
        # s = [Variable(ss) for ss in s] if isinstance(s, list) else Variable(s)
        # g = [Variable(gg) for gg in g] if isinstance(g, list) else Variable(g)

        if isinstance(s, list):
            out = self.net.forward(*s)
        else:
            out = self.net.forward(s)
        _pairs = zip(out.flatten().data.cpu(), g.flatten().data.cpu(), torch.sigmoid(out).flatten().data.cpu())
        _df = pd.DataFrame(_pairs, columns=['res', 'g', 'sig_res'], dtype=float)
        self._logger.debug('\n' + _df.to_string())
        del _pairs, _df
        return out

    def _loss_eval(self, *args):
        out, s, g = args
        if self.iscuda:
            g = self._force_cuda(g)

        loss = self.lossfunction(out.squeeze(), g.squeeze().long())
        return loss

    def validation(self):
        if self._data_loader_val is None:
            self._logger.warning("Validation skipped because no loader is available.")
            return []
        with torch.no_grad():
            self.net.eval()

            decisions = []
            validation_loss = []
            for s, g in tqdm(self._data_loader_val, desc="Validation", position=2):
                if self.iscuda:
                        s = [ss.cuda() for ss in s] if isinstance(s, list) else s.cuda()
                        g = [gg.cuda() for gg in g] if isinstance(g, list) else g.cuda()

                if isinstance(s, list):
                    res = self.net(*s)
                else:
                    res = self.net(s)
                # res = torch.(res, dim=1)
                while res.dim() < 2:
                    res = res.unsqueeze(0)
                dic = torch.argmax(torch.softmax(res, dim=1), dim=1)
                decisions.extend([guess == truth for guess, truth in zip(dic.tolist(), g.tolist())])
                loss = self._loss_eval(res, s, g)
                validation_loss.append(loss.item())

            # Compute accuracies
            acc = float(decisions.count(True)) / float(len(decisions))
            validation_loss = np.mean(np.array(validation_loss).flatten())
            self._logger.log_print_tqdm("Validation Result - ACC: %.05f, VAL: %.05f"%(acc, validation_loss))

        self.plotter_dict['scalars']['Loss/Validation Loss'] = validation_loss
        self.plotter_dict['scalars']['Performance/ACC'] = acc
        return validation_loss, acc

