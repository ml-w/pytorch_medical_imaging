import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .SolverBase import SolverBase, SolverBaseCFG

from typing import Optional, Iterable


class SegmentationSolverCFG(SolverBaseCFG):
    r"""Configuration for :class:`SegmentationSolver`

    Class Attributes:
        [Required] class_weights (torch.Tensor):
            Define the class weights passed to the loss function. If set to 0, no weigths will be used.
        sigmoid_params (dict, Optional):
            By default the class weights are adjusted using a sigmoid function during training to improve the training
            efficiency and also the segmentation performance. See :func:`decay_optimizer` for more details.
        decay_init_epoch (int, Optional):
            Used by the :func:`decay_optimizer` in case the solver is not starting from epoch 0. Default to 0.

    """
    class_weights : Iterable[float] = None
    sigmoid_params: Optional[dict]  = dict(
        delay = 15,
        stretch = 2,
        cap = 0.3
    )
    decay_init_epoch: Optional[int] = 0



class SegmentationSolver(SolverBase):
    def __init__(self, cfg: SegmentationSolverCFG, *args, **kwargs):
        r"""This solver trains segmentation networks, but can also be used for image2image networks if you work with
        the loss function yourself. For attributes and settings, see :class:`SegmentationSolverCFG`.

        Args:
            cfg (SegmentationSolverCFG):
                Configurations.

        See Also:
            * :class:`SegmentationSolverCFG`
            * :class:`SolverBaseCFG`


        """
        super(SegmentationSolver, self).__init__(cfg, *args, **kwargs)


    # def prepare_lossfunction(self):
    #     if self.class_weights is None:
    #         self._logger.warning("Automatic computing weighs are not supported now!")
    #         raise DeprecationWarning("Automatic computing weighs are not supported now!")
    #
    #     # set class weights to 0 to disable class weight for loss function
    #     if not self.class_weights == 0:
    #         weights = torch.as_tensor(self.class_weights)
    #         loss_init_weights = weights.cpu().float()
    #         self._logger.info("Initial weight factor: " + str(weights))
    #     else:
    #         self._logger.info("Skipping class weights.")
    #         loss_init_weights = None
    #
    #     if self.loss_function is None:
    #         self.loss_function = nn.CrossEntropyLoss(weight=loss_init_weights)
    #     else:
    #         self._logger.warning("Loss function is already created.")
    #
    #         # make sure the weight tensor is float
    #         if not self.loss_function.weight.dtype.is_floating_point:
    #             self._logger.warning(f"Loss function weight is not of type `float`, trying to re-cast it.")
    #             self.loss_function.weight = self.loss_function.weight.float()

    def auto_compute_class_weights(self,
                                   gt_data: torch.Tensor,
                                   ) -> torch.Tensor:
        r"""Compute the counts of each class in the ground-truth data. Use for class weights in optimizer.

        Class weight is computed based on the reciprocal of its counts, normalized linearly such that the class with
        the fewest counts has a weight of 1. Note that the null class is ramped up by a fixed factor to ensure its
        weight is not too low for training.

        .. math::

            \text{weight}(c) = \frac{N_{cmin}}{N_c}

        .. note::
            This is not the best way to calculate the class weights, normally you want to calculate it based on all
            the labels rather than just one data point. It is therefore recommended to set your own weights always.


        Returns:
            torch.Tensor: A tensor with the same number of element as the total number of unique class in ``gt_data``.

        """
        self._logger.info("Detecting number of classes...")
        values, counts = gt_data.unique(return_counts=True)
        valcountpair = {k.item(): v.item() for k, v in zip(values, counts)}
        classes = list(values())
        classes.sort()
        numOfClasses = len(values)
        self._logger.info("Find %i classes: %s" % (numOfClasses, classes))

        if numOfClasses == 1:
            msg = "Your first target contains no labels!"
            raise ArithmeticError(msg)

        # calculate empty label ratio for updating loss function weight
        min_count = counts.min()
        r = float(min_count)/counts.numpy().astype('float')
        del valcountpair  # free RAM

        # calculate init-factor for sigmoid weight scheduling
        if not self.class_weights is None:
            weights = self.sigmoid_plus(self.decay_init_epoch + 1,
                                        r,
                                        self.sigmoid_params['stretch'],
                                        self.sigmoid_params['delay'],
                                        self.sigmoid_params['cap'])

        # null class can't be too low
        weights[0] *= 10
        return weights

    def _validation_callback(self):
        r"""Compute the DICE of the validation loop. Also put stuff into :attr:`plotter_dict` for plotting to
        tensorboard.
        """
        tps, tns, fps, fns = torch.tensor(self.perfs).sum(dim=0)
        dsc = self._DICE(tps, fps, tns, fns)
        mean_val_loss = np.mean(np.array(self.validation_losses).flatten())
        self._logger.info("Validation Result VAL: %.05f DSC: %.05f" % (mean_val_loss, dsc))
        self.plotter_dict['scalars']['Loss/Validation Loss'] = mean_val_loss
        self.plotter_dict['scalars']['Perf/Validation DSC'] = dsc

    def _validation_step_callback(self, g, res, loss, uids=None):
        r"""Count the FP, TN, FP and FN.

        Attributes:
            validation_losses (Iterable[float]):
                List storing the losses of each step.
            perf (Iterable[Any]):
                List storing data need to calculated the performance.

        Args:
            uids:
            g (torch.Tensor):
                Label tensor.
            res (torch.Tensor):
                Network output tensor.
            loss (torch.Tensor or float):
                Loss of the step.
        """
        self.validation_losses.append(loss.detach().cpu().data.item())
        # Compute hit and misses
        res_b = res.argmax(dim=1)
        res_b = res_b.bool().flatten().detach().cpu()
        fg = g.bool().flatten().detach().cpu()
        tp = (res_b * fg).int().sum().float().cpu().item()
        tn = (~res_b * ~fg).int().sum().float().cpu().item()
        fp = (res_b * ~fg).int().sum().float().cpu().item()
        fn = (~res_b * fg).int().sum().float().cpu().item()
        self.perfs.append([tp, tn, fp, fn])

    def _loss_eval(self, *args):
        out, s, g = args
        g   = self._match_type_with_network(g)
        out = self._match_type_with_network(out)

        if isinstance(self.loss_function, (nn.BCELoss, nn.BCEWithLogitsLoss)):
            out = F.log_softmax(out, dim=1)
            if not g.dim() == out.dim():
                g = g.squeeze()
            loss = self.loss_function(out, g.long())
        else:
            # For other loss function, deal with the dimension yourselves
            if g.dim() == 5 and g.shape[1] == 1:
                g = g.squeeze()
            loss = self.loss_function(out, g.long())
        return loss

    def decay_optimizer(self, *args):
        r"""
        Function is called every epoch, the weight of the optimizer is fine tuned such that initially the loss
        weights are balanced by the count of each class, the weight will quickly shift towards favoring empty pixel
        for the first dozens of epoch until the all weights slowly converge to 1. See the actual function in
        :func:`sigmoid_plus`.

        Args:
            *args: Pass to superclass method

        Returns:

        .. note::
            TODO:
                * Allow loading parameters.

        .. note::
            Currently the coefficient :math:`s` and :math:`d` are set in attribute `_sigmoid_params`

        See Also:
            * :func:`sigmoid_plus`

        """
        super().decay_optimizer(*args)

        # Decay the class weight using a sigmoid curve.
        try:
            s = self.sigmoid_params['stretch']
            d = self.sigmoid_params['delay']
            cap = self.sigmoid_params['cap']
            if isinstance(self.loss_function, nn.CrossEntropyLoss):
                self._logger.debug('Current weight: ' + str(self.loss_function.weight))
                offset = self._decayed_time + self.decay_init_epoch # init_weight is t_0
                new_weight = torch.as_tensor([self.sigmoid_plus(offset, self.class_weights[i], s, d, cap) \
                                              for i in range(len(self.class_weights))])
                self.loss_function.weight.copy_(new_weight)
                self._logger.debug('New weight: ' + str(self.loss_function.weight))
        except (AttributeError, KeyError):
            msg = "Sigmoid param was not provided, skipping loss weight scheduling."
            self._logger.warning(msg, no_repeat=True)


    @staticmethod
    def sigmoid_plus(x, init, stretch, delay, cap) -> float:
        r"""Sigmoid function to alter class weights

        .. math::

            s(x;x_0, s, d) = x_0 + \frac{(1 - x_0)}{(1 + e^{-(x - 2d) / s}}

        Where:

        +--------------+----------------------------------------------------------------------------------------------+
        | Name         | Description                                                                                  |
        +==============+==============================================================================================+
        | :math:`x`    | Input weight.                                                                                |
        +--------------+----------------------------------------------------------------------------------------------+
        | :math:`x_0`  | Initial count, used when training is not starting from 0th epoch, say for after loading cp.  |
        +--------------+----------------------------------------------------------------------------------------------+
        | :math:`s`    | Stretch, controls the slope of the convergence                                               |
        +--------------+----------------------------------------------------------------------------------------------+
        | :math:`d`    | Delay, controls which epochs starts the convergence                                          |
        +--------------+----------------------------------------------------------------------------------------------+

        Returns:
            float

        """
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
    def _DICE(TP, FP, TN, FN) -> None:
        r"""Retrun the DICE score.

        .. math::

            \text{DICE} = \frac{2\cdot TP}{2 * TP + FP + FN}

        Returns:
            float: Dice score :math:`\in [0, 1]`

        """
        if np.isclose(2*TP+FP+FN, 0):
            return 1
        else:
            return 2*TP / (2*TP+FP+FN)


    def _step_callback(self, s, g, out, loss, step_idx=None):
        r"""Plot segmentation. Requires the presence of a ``tb_plotter`` attribute."""
        if self._tb_plotter is None:
            self._logger.warning("There are no tb_plotter.", True)
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

    def _epoch_callback(self, *args, **kwargs):
        self.decay_optimizer()
        super(SegmentationSolver, self)._epoch_callback(*args, **kwargs)
