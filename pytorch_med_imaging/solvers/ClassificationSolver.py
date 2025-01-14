from .SolverBase import SolverBase, SolverBaseCFG
from mnts.mnts_logger import MNTSLogger


from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np
from tqdm import *
from ..loss import CumulativeLinkLoss
from typing import Optional, Union, Iterable, Any

import pandas as pd
import pprint


class ClassificationSolverCFG(SolverBaseCFG):
    r"""Configuration file for initializing :class:`ClassificationSolver`.

    Class Attributes:
        [Required] class_weights (torch.Tensor):
            Define the class weights passed to the loss function. If set to 0, no weigths will be used.
        sigmoid_params (dict, Optional):
            By default the class weights are adjusted using a sigmoid function during training to improve the training
            efficiency and also the segmentation performance. See :func:`decay_optimizer` for more details.
        decay_init_epoch (int, Optional):
            Used by the :func:`decay_optimizer` in case the solver is not starting from epoch 0. Default to 0.
        ordinal_class (bool, Optional):
            This option is for special loss function that improves cardinal classification. Behavior is slightly
            changed due to syntax requirements if this is set to ``True``. Default to ``False``.
        ordinal_mse (bool, Optional):
            This option is for special loss function that improves cardinal classification. Behavior is slightly
            changed due to syntax requirements if this is set to ``True``. Default to ``False``.
        sig_out (bool, Optional):
            This option is for telling if sigmoid is required for processing network output. Default to ``False``.

    """
    class_weights : Iterable[float] = None
    sigmoid_params: Optional[dict]  = dict(
        delay = 15,
        stretch = 2,
        cap = 0.3
    )
    decay_init_epoch: Optional[int]  = 0
    ordinal_class   : Optional[bool] = False
    ordinal_mse     : Optional[bool] = False
    sig_out         : Optional[bool] = False

class ClassificationSolver(SolverBase):
    def __init__(self, cfg, *args, **kwargs):
        r"""Solver for classification tasks. For details to kwargs, see :class:`SolverBase`.

        Args:
            cfg *ClassificationSolverCFG):
                Configuration.

        Attributes:
            _validation_misclassification_record (dict):
                Record of misclassifications during validation. This dictionary is populated in the method
                :func:`_validation_step_callback` and is only populated when the argument `uid` is provided and is a
                hashable object.

        """
        super(ClassificationSolver, self).__init__(cfg, *args, **kwargs)
        self._validation_misclassification_record = {}

    def _build_validation_df(self, g, res, uid=None):
        r"""Build pandas table for displaying the prediction results. Displayed after each step and is called in
        the step callback.
        """
        g, res = g.cpu().detach(), res.cpu().detach()
        _df = pd.DataFrame.from_dict({f'res_{d}': list(res[:, d].numpy())
                                      for d in range(res.shape[-1])})
        if not self.ordinal_mse:
            if g.dim() == 1:
                _df_gt = pd.DataFrame.from_dict({'gt': list(g.flatten().numpy())})
                _df = pd.concat([_df, _df_gt], axis=1)
                _df['predicted'] = torch.argmax(res.squeeze(), dim=1).numpy()
                _df['eval'] = (_df['predicted'] == _df['gt']).replace({True: 'Correct', False: 'Wrong'})
            else:
                _df_gt = pd.DataFrame.from_dict({f'gt_{d}': list(g[:, d].numpy())
                                                 for d in range(g.shape[-1])})
                _df = pd.concat([_df, _df_gt], axis=1)
        else:
            _df_gt = pd.DataFrame.from_dict({'gt': list(g.flatten().numpy())})
            _df = pd.concat([_df, _df_gt], axis=1)
            _df['predicted'] = torch.round(res.squeeze()).long().numpy()
            _df['eval'] = (_df['predicted'] == _df['gt']).replace({True: 'Correct', False: 'Wrong'})

        if not uid is None:
            _df.index = uid
        return _df, _df['predicted']

    def _loss_eval(self, *args):
        r"""Inherit this to handle usage of :class:`CumulativeLinkLoss` and also using MSE as loss function for ordinal
        classification situations.

        Args:
            *args:
                Expect to the [s, g].

        See Also:
            * :class:`CumulativeLinkLoss`
        """
        out, s, g = args

        s = self._match_type_with_network(s)
        g = self._match_type_with_network(g)

        g, out = self._align_g_res_size(g, out)

        if self.ordinal_class:
            if not isinstance(self.loss_function, CumulativeLinkLoss):
                msg = f"For oridinal_class mode, expects `CumulativeLinkLoss` as the loss function, got " \
                      f"{type(self.loss_function)} instead."
                raise AttributeError(msg)

        if self.ordinal_mse and not isinstance(self.loss_function, nn.SmoothL1Loss):
                msg = f"For oridinal_mse mode, expects `SmoothL1Loss` as the loss function, got " \
                      f"{type(self.loss_function)} instead."
                raise AttributeError(msg)

        # required dimension of CrossEntropy is either (B) or (B, num_class)
        if isinstance(self.loss_function, nn.CrossEntropyLoss):
            # squeeze (B, 1) to (B)
            g = g.squeeze()

        self._logger.debug(f"Output size out: {out.shape}({out.dtype}) g: {g.shape}({g.dtype})")
        # Cross entropy does not need any processing, just give the raw output
        loss = self.loss_function(out, g)
        return loss

    def _align_g_res_size(self, g, res):
        """For ordinary classification, expects network output `res` and target labels `g` dimension to be
        :math:`(B × C)` and :math:`(B × 1)` where :math:`C` is the number of classes in label. For binary classification
        here, we allow users to ask more than one binary questions such that the network output and the labels should
        both have a dimension of :math:`(B × C)` where :math:`C` is the number of questions asked.

        Generally speaking, it is common to have :math:`C = 1`, but the dimension 1 poses trouble because it gets
        squeezed by calling `torch.Tensor.squeeze`.

        Args:
            g (torch.Tensor):
                The target label tensor. Should have a dimension of :math:`(B)`, but if not, it gets reshaped.
            res (torch.Tensor):
                The network output tensor. Should have a dimension of :math:`(B × C)`.

        Returns:
            torch.Tensor: The reshaped target label. The reshaped output tensor.

        """
        res = res.squeeze()  # Expect (B x C) where C is same as number of classes
        # If ordinal_mse mode, assume loss function is SmoothL1Loss
        if g.squeeze().dim() == 1 and not self.ordinal_mse:
            g = g.squeeze().long()
        return g, res

    def _validation_step_callback(self, g: torch.Tensor, res: torch.Tensor, loss: Union[torch.Tensor, float],
                                  uids: Union[Iterable, None]=None) -> None:
        r"""Stores decision, step loss and performance.

        Args:
            g (torch.Tensor):
                Expect input to be :math:`(B)`
            res (torch.Tenor):
                Expect input to be :math:`(B\times \texttt{num of class})`
            loss (torch.Tensor or float):
                Should be a single value, or a tensor with a single value.
            uids (list, Iterable):
                Expect length identical as `g`

        Returns:
            None
        """
        self.validation_losses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
        # when ordinal_mse mode, the decision is based on rounding the probability to the nearest class.
        if not self.ordinal_mse:
            dic = torch.argmax(res, dim=1)
        else:
            dic = torch.round(res).long()
        # add the classification result to the list
        self.perfs.extend([guess == truth for guess, truth in zip(dic.tolist(), g.tolist())])
        # add the misclassified result to the dictionary
        self._update_misclassification_record(dic, g, uids)

    def _update_misclassification_record(self, dic, g, uids=None):
        r"""Updates the misclassification record for the given data and logs it.

        This function takes in the predicted classifications, true classifications, and unique
        identifiers of data samples. It updates the misclassification record by incrementing
        the count for each misclassified data sample. Finally, it logs the updated
        misclassification record.

        Args:
            dic (array-like):
                The array of predicted classifications.
            g (array-like):
                The array of true classifications.
            uids (list, optional):
                A list of unique identifiers for the data samples. If not
                provided, the misclassification record will not be updated.

        Returns:
            None
        """
        if uids is not None:
            # find out which element is wrong
            wrong_pred = {idx: guess != truth for idx, guess, truth in zip(uids, dic.tolist(), g.tolist())}
            for idx, miss_classififed in wrong_pred.items():
                if not miss_classififed:
                    continue
                if idx in self._validation_misclassification_record:
                    self._validation_misclassification_record[idx] += 1
                else:
                    self._validation_misclassification_record[idx] = 1

    def _validation_callback(self) -> None:
        r"""Calculate accuracy of classification.
        """
        # Compute accuracies
        acc = float(self.perfs.count(True)) / float(len(self.perfs))
        self.validation_losses = np.mean(np.array(self.validation_losses).flatten())
        self._logger.info("Validation Result - ACC: %.05f, VAL: %.05f"%(acc, self.validation_losses))
        self.plotter_dict['scalars']['val/loss'] = self.validation_losses
        self.plotter_dict['scalars']['val/performance/ACC'] = acc

        # Print the misclassification report
        if len(self._validation_misclassification_record) > 0:
            self._logger.info("Validation misclassification report: {}".format(
                pprint.pformat(self._validation_misclassification_record)
            ))


    def _step_callback(self, s, g, out, loss, uid=None, step_idx=None) -> None:
        r"""Build and print a table summarizing the prediction of the step.

        Args:
            uid:
            s (torch.Tensor)            : The network input of the step.
            g (torch.Tensor)            : The target label of the step.
            out (torch.Tensor)          : The network output.
            loss (float or torch.Tensor): The loss.
            uid (list of str)           : The uids.
            step_idx (int, Optional)    : The number of steps.
        """
        # Print step information
        self._logger.debug(f"s: {s.shape} out: {out.shape}")
        _df, _ = self._build_validation_df(g, out, uid=uid)
        self._logger.debug('\n' + _df.to_string())

        # These are used for specific network and will be move to other places soon.
        if hasattr(self.net, 'module'):
            if hasattr(self.net.module, '_batch_callback'):
                self.net.module._batch_callback()
                self._logger.debug(f"LCL:{self.net.module.LCL.cutpoints}")
        elif hasattr(self.net, '_batch_callback'):
            self.net.module._batch_callback()
            self._logger.debug(f"LCL:{self.net.module.LCL.cutpoints}")