import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from .SolverBase import SolverBase
from ..med_img_dataset import ImageDataSet, ImageDataMultiChannel, PMIDataBase
from mnts.mnts_logger import MNTSLogger

import tqdm.auto as auto
import torchio as tio

__all__ = ['SegmentationSolver']

class SegmentationSolver(SolverBase):
    def __init__(self,
                 net, hyperparameters, use_cuda):
        r"""This solver trains segmentation networks. The following items n

        Attributes:
            solverparams_sigmoid_params (dict):
                Default to {'delay': 15, 'stretch': 2, 'cap': 0.3}.
            solverparams_init_weight (int):
                Default to 0.
            solverparams_class_weights (float):
                Weight of each class used in lossfunction. Default to None.
            solverparams_decay_init_weight (float):
                If the training is not started at epoch 0, set this to continue the curve designed using
                the :func:`sigmoid_plus`.
        """
        super(SegmentationSolver, self).__init__(net, hyperparameters, use_cuda)

    def _load_default_attr(self, _):
        r"""Inherit this to get more default hyperparameters"""
        default_attr = {
            'solverparams_sigmoid_params'   : {'delay': 15, 'stretch': 2, 'cap': 0.3},
            'solverparams_class_weights'    : None,
            'solverparams_decay_init_weight': 0
        }
        super(SegmentationSolver, self)._load_default_attr(default_attr)

    def create_lossfunction(self):
        if self.solverparams_class_weights is None:
            self._logger.warning("Automatic computing weigths are not supported now!")
            raise DeprecationWarning("Automatic computing weigths are not supported now!")

        # set class weights to 0 to disable class weight for loss function
        if not self.solverparams_class_weights == 0:
            weights = torch.as_tensor(self.solverparams_class_weights)
            loss_init_weights = weights.cpu().float()
            self._logger.info("Initial weight factor: " + str(weights))
        else:
            self._logger.info("Skipping class weights.")
            loss_init_weights = None
        self.lossfunction = nn.CrossEntropyLoss(weight=loss_init_weights) #TODO: Allow custom loss function

    def auto_compute_class_weights(self, gt_data, param_initWeight):
        r"""Compute the counts of each class in the ground-truth data. Use for class weights in optimizer."""
        raise NotImplementedError

        self._logger.info("Detecting number of classes...")
        valcountpair = gt_data.get_unique_values_n_counts()
        classes = list(valcountpair.keys())
        numOfClasses = len(classes)
        self._logger.info("Find %i classes: %s" % (numOfClasses, classes))

        # calculate empty label ratio for updating loss function weight
        r = []
        for c in classes:
            factor = float(np.prod(np.array(gt_data.size()))) / float(valcountpair[c])
            r.append(factor)
        r = np.array(r)
        r = r / r.max()
        self.solverparams_initial_weight = r
        del valcountpair  # free RAM

        # calculate init-factor for sigmoid weight scheduling
        if not param_initWeight is None:
            self.solverparams_initial_weight = self.sigmoid_plus(param_initWeight + 1, self.solverparams_initial_weight, self.sigmoid_params['stretch'],
                                                                 self.sigmoid_params['delay'], self.sigmoid_params['cap'])

        # null class can't be too low
        self.solverparams_initial_weight[0] = self.solverparams_initial_weight[0] * 10
        return

    def validation(self) -> list:
        if self._data_loader_val is None:
            self._logger.warning("Validation skipped because no loader is available.")
            return None

        with torch.no_grad():
            validation_loss = []
            perfs = []
            self.net.eval()
            for mb in auto.tqdm(self._data_loader_val, desc="Validation", position=2):
                s, g = self._unpack_minibatch(mb, self.solverparams_unpack_keys_forward)
                s = self._match_type_with_network(s)
                g = self._match_type_with_network(g) # no assumption but should be long in segmentation only.

                if isinstance(s, list):
                    res = self.net.forward(*s)
                else:
                    res = self.net.forward(s)
                res = F.log_softmax(res, dim=1)
                loss = self.lossfunction(res, g.squeeze().long())
                validation_loss.append(loss.detach().cpu().data.item())
                self._logger.debug("_val_step_loss: {}".format(loss.cpu().data.item()))

                # Compute hit and misses
                res_b = res.argmax(dim=1)
                res_b = res_b.bool().flatten().detach().cpu()
                fg = g.bool().flatten().detach().cpu()

                tp = (res_b * fg).int().sum().float().cpu().item()
                tn = (~res_b * ~fg).int().sum().float().cpu().item()
                fp = (res_b * ~fg).int().sum().float().cpu().item()
                fn = (~res_b * fg).int().sum().float().cpu().item()
                perfs.append([tp, tn, fp, fn])
                del mb, s, g, res_b, fg, tp, tn, fp, fn, loss
                gc.collect()

            tps, tns, fps, fns = torch.tensor(perfs).sum(dim=0)
            dsc = self._DICE(tps, fps, tns, fns)
            mean_val_loss = np.mean(np.array(validation_loss).flatten())
            self._logger.info("Validation Result VAL: %.05f DSC: %.05f"%(mean_val_loss, dsc))
        self.net = self.net.train()

        self.plotter_dict['scalars']['Loss/Validation Loss'] = mean_val_loss
        self.plotter_dict['scalars']['Perf/Validation DSC'] = dsc
        return mean_val_loss

    def _feed_forward(self, *args):
        s, g = args
        try:
            s = self._match_type_with_network(s)
        except Exception as e:
            self._logger.exception("Failed to match input to network type. Falling back.")
            raise RuntimeError("Feed forward failure") from e

        if isinstance(s, list):
            out = self.net.forward(*s)
        else:
            out = self.net.forward(s)
        return out

    def _loss_eval(self, *args):
        out, s, g = args
        if (isinstance(g, list) or isinstance(g, tuple)) and len(g) > 1:
            g = [Variable(gg, requires_grad=False) for gg in g]
        else:
            g = Variable(g, requires_grad=False)

        if self.iscuda:
            g = self._force_cuda(g)

        loss = self.lossfunction(out, g.squeeze().long())
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

        # Decay the class weight using a sigmoid curve.
        s = self.solverparams_sigmoid_params['stretch']
        d = self.solverparams_sigmoid_params['delay']
        cap = self.solverparams_sigmoid_params['cap']
        if isinstance(self.lossfunction, nn.CrossEntropyLoss):
            self._logger.debug('Current weight: ' + str(self.lossfunction.weight))
            offset = self._decayed_time + self.solverparams_decay_init_weight # init_weight is t_0
            new_weight = torch.as_tensor([self.sigmoid_plus(offset, self.solverparams_class_weights[i], s, d, cap) for i in range(len(
                self.solverparams_class_weights))])
            self.lossfunction.weight.copy_(new_weight)
            self._logger.debug('New weight: ' + str(self.lossfunction.weight))


    @staticmethod
    def sigmoid_plus(x, init, stretch, delay, cap):
        r"""Sigmoid function to alter class weights"""
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

        # sometimes there extra inputs provided to forward, conventionally, the first input should be the image
        # therefore, use image for plotting
        if isinstance(s, (tuple, list)):
            s = s[0]

        if step_idx % 10 == 0:
            # make sure they are not remaining in the gpu.
            self._tb_plotter.plot_segmentation(g.cpu(), out.cpu(), s.cpu().float(), step_idx)

        # delete references
        del s, g, out