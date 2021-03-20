import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from .SolverBase import SolverBase
from pytorch_med_imaging.logger import Logger

import tqdm.auto as auto

__all__ = ['SegmentationSolver']

class SegmentationSolver(SolverBase):
    def __init__(self, in_data, gt_data, net, param_optim, param_iscuda,
                 param_initWeight=None, logger=None, config=None):
        assert isinstance(logger, Logger) or logger is None, "Logger incorrect settings!"

        if logger is None:
            logger = Logger[self.__class__.__name__]

        self._decay_init_weight = param_initWeight if not param_initWeight is None else 0

        solver_configs = {}
        # check unique class in gt
        logger.info(f"gt_data: {gt_data}")
        logger.log_print_tqdm("Detecting number of classes...")
        valcountpair = gt_data.get_unique_values_n_counts()
        classes = list(valcountpair.keys())
        numOfClasses = len(classes)
        logger.log_print_tqdm("Find %i classes: %s"%(numOfClasses, classes))

        # calculate empty label ratio for updating loss function weight
        self._sigmoid_params = {'delay': 15, 'stretch': 2, 'cap':0.3}
        r = []
        for c in classes:
            factor = float(np.prod(np.array(gt_data.size())))/float(valcountpair[c])
            r.append(factor)
        r = np.array(r)
        r = r / r.max()
        self._r = r
        del valcountpair # free RAM

        # calculate init-factor
        if not param_initWeight is None:
            self._r = self.sigmoid_plus(param_initWeight + 1, self._r, self._sigmoid_params['stretch'],
                                       self._sigmoid_params['delay'], self._sigmoid_params['cap'])

        # null class can't be too low
        self._r[0] = self._r[0] * 10
        weights = torch.as_tensor(self._r)
        self.loss_init_weights = weights.cpu().float()
        logger.log_print_tqdm("Initial weight factor: " + str(weights))

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
            net = net(inchan, numOfClasses)

        # Create optimizer and loss function
        lossfunction = nn.CrossEntropyLoss(weight=self.loss_init_weights) #TODO: Allow custom loss function
        optimizer = optim.SGD(net.parameters(), lr=param_optim['lr'], momentum=param_optim['momentum'])
        # optimizer = optim.Adam(net.parameters(), lr=param_optim['lr'])
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

    def validation(self):
        if self._data_loader_val is None:
            self._logger.warning("Validation skipped because no loader is available.")
            return []

        with torch.no_grad():
            validation_loss = []
            perfs = []
            self._net.eval()
            for s, g in auto.tqdm(self._data_loader_val, desc="Validation", position=2):
                s = self._match_type_with_network(s)
                g = self._match_type_with_network(g) # no assumption but should be long in segmentation only.

                if isinstance(s, list):
                    res = self._net(*s)
                else:
                    res = self._net(s)
                res = F.log_softmax(res, dim=1)
                loss = self._lossfunction(res, g.squeeze().long())
                validation_loss.append(loss.item())
                self._logger.debug("_val_step_loss: {}".format(loss.data.item()))

                # Compute hit and misses
                res_b = res.argmax(dim=1)
                res_b = res_b.bool().flatten()
                fg = g.bool().flatten()

                tp = (res_b * fg).int().sum().float().cpu().item()
                tn = (~res_b * ~fg).int().sum().float().cpu().item()
                fp = (res_b * ~fg).int().sum().float().cpu().item()
                fn = (~res_b * fg).int().sum().float().cpu().item()
                perfs.append([tp, tn, fp, fn])

            tps, tns, fps, fns = torch.tensor(perfs).sum(dim=0)
            dsc = self._DICE(tps, fps, tns, fns)
            mean_val_loss = np.mean(np.array(validation_loss).flatten())
            self._logger.info("Validation Result VAL: %.05f DSC: %.05f"%(mean_val_loss, dsc))
        self._net = self._net.train()

        self.plotter_dict['scalars']['Loss/Validation Loss'] = mean_val_loss
        self.plotter_dict['scalars']['Perf/Validation DSC'] = dsc
        return [mean_val_loss]

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
            Currently the coefficient :math:`s` and :math:`d` are set in attribute `_sigmoid_params`

        """
        super().decay_optimizer(*args)

        s = self._sigmoid_params['stretch']
        d = self._sigmoid_params['delay']
        cap = self._sigmoid_params['cap']
        if isinstance(self._lossfunction, nn.CrossEntropyLoss):
            self._logger.log_print_tqdm('Current weight: ' + str(self._lossfunction.weight), 20)
            offset = self._decayed_time + self._decay_init_weight
            new_weight = torch.as_tensor([self.sigmoid_plus(offset, self._r[i], s, d, cap) for i in range(len(
                self._r))])
            self._lossfunction.weight.copy_(new_weight)
            self._logger.log_print_tqdm('New weight: ' + str(self._lossfunction.weight), 20)


    @staticmethod
    def sigmoid_plus(x, init, stretch, delay, cap):
        # sigmoid increase to cap.
        out = (init + (cap - init) * 1. / (1 + np.exp(- x / stretch + delay * 2 / stretch)))

        # no decrease.
        try:
            out[np.array(init) > cap] = init[np.array(init) > cap]
        except TypeError:
            if init > cap:
                out = init
        return out

    @staticmethod
    def _perf_measure(y_guess, y_actual):
        """
        Obtain the result of index test, i.e. the TF, FP, TN and FN of the test.

        Args:
            y_actual (np.array): Actual class.
            y_guess (np.array): Guess class.

        Returns:
            (list of int): Count of TP, FP, TN and FN respectively
        """

        y = y_actual.astype('bool').flatten()
        x = y_guess.astype('bool').flatten()

        TP = np.sum((y == True) & (x == True))
        TN = np.sum((y == False) & (x == False))
        FP = np.sum((y == False) & (x == True))
        FN = np.sum((y == True) & (x == False))
        TP, TN, FP, FN = [float(v) for v in [TP, TN, FP, FN]]
        return TP, FP, TN, FN

    @staticmethod
    def _DICE(TP, FP, TN, FN):
        if np.isclose(2*TP+FP+FN, 0):
            return 1
        else:
            return 2*TP / (2*TP+FP+FN)


    def _step_callback(self, s, g, out, loss, step_idx=None):
        if self._tb_plotter is None:
            self._logger.warning("There are no tb_plotter.")
            return

        if step_idx % 10 == 0:
            self._tb_plotter.plot_segmentation(g, out, s, step_idx)