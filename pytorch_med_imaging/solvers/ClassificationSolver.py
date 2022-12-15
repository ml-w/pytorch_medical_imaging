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

class ClassificationSolver(SolverBase):
    def __init__(self, cfg, *args, **kwargs):
        r"""Solver for classification tasks. For details to kwargs, see :class:`SolverBase`.

        Args:
            cfg *ClassificationSolverCFG):
                Configuration.
        """
        super(ClassificationSolver, self).__init__(cfg, *args, **kwargs)

    def _build_validation_df(self, g, res, uid=None):
        r"""Build pandas table for displaying the prediction results. Displayed after each step and is called in
        the step callback.
        """
        _df = pd.DataFrame.from_dict({f'res_{d}': list(res[:, d].cpu().detach().numpy())
                                      for d in range(res.shape[-1])})
        if not self.ordinal_mse:
            if g.dim() == 1:
                _df_gt = pd.DataFrame.from_dict({'gt': list(g.flatten().cpu().detach().numpy())})
                _df = pd.concat([_df, _df_gt], axis=1)
                _df['predicted'] = torch.argmax(res.squeeze(), dim=1).cpu().detach().numpy()
                _df['eval'] = (_df['predicted'] == _df['gt']).replace({True: 'Correct', False: 'Wrong'})
            else:
                _df_gt = pd.DataFrame.from_dict({f'gt_{d}': list(g[:, d].cpu().detach().numpy())
                                                 for d in range(g.shape[-1])})
                _df = pd.concat([_df, _df_gt], axis=1)
        else:
            _df_gt = pd.DataFrame.from_dict({'gt': list(g.flatten().cpu().detach().numpy())})
            _df = pd.concat([_df, _df_gt], axis=1)
            _df['predicted'] = torch.round(res.squeeze()).cpu().detach().long().numpy()
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

        if self.use_cuda:
            s = self._match_type_with_network(s)
            g = self._match_type_with_network(g)

        out = out.squeeze() # Expect (B x C) where C is same as number of classes

        # If ordinal_mse mode, assume loss function is SmoothL1Loss
        if g.squeeze().dim() == 1 and not self.ordinal_mse:
            g = g.squeeze().long()

        if self.ordinal_class:
            if not isinstance(self.loss_function, CumulativeLinkLoss):
                msg = f"For oridinal_class mode, expects `CumulativeLinkLoss` as the loss function, got " \
                      f"{type(self.loss_function)} instead."
                raise AttributeError(msg)

        if self.ordinal_mse and not isinstance(self.loss_function, nn.SmoothL1Loss):
                msg = f"For oridinal_mse mode, expects `SmoothL1Loss` as the loss function, got " \
                      f"{type(self.loss_function)} instead."
                raise AttributeError(msg)

        self._logger.debug(f"Output size out: {out.shape} g: {g.shape}")
        # Cross entropy does not need any processing, just give the raw output
        loss = self.loss_function(out, g)
        return loss

    def validation(self):
        r"""Validation for classification problems.

        Returns:
            float: Validation loss.
        """
        if self.data_loader_val is None:
            self._logger.warning("Validation skipped because no loader is available.")
            return None
        with torch.no_grad():
            self.net.eval()

            decisions = []
            validation_loss = []
            for mb in tqdm(self.data_loader_val, desc="Validation", position=2):
                s, g = self._unpack_minibatch(mb, self.unpack_key_forward)
                s = self._match_type_with_network(s)
                g = self._match_type_with_network(g)

                if isinstance(s, list):
                    res = self.net(*s)
                else:
                    res = self.net(s)
                # res = torch.(res, dim=1)
                while res.dim() < 2:
                    res = res.unsqueeze(0)

                # when ordinal_mse mode, the decision is based on rounding the probability to the nearest class.
                if not self.ordinal_mse:
                    dic = torch.argmax(torch.softmax(res, dim=1), dim=1)
                else:
                    dic = torch.round(res).long()
                decisions.extend([guess == truth for guess, truth in zip(dic.tolist(), g.tolist())])
                loss = self._loss_eval(res, s, g)
                validation_loss.append(loss.item())

            # Compute accuracies
            acc = float(decisions.count(True)) / float(len(decisions))
            validation_loss = np.mean(np.array(validation_loss).flatten())
            self._logger.info("Validation Result - ACC: %.05f, VAL: %.05f"%(acc, validation_loss))

        self.plotter_dict['scalars']['Loss/Validation Loss'] = validation_loss
        self.plotter_dict['scalars']['Performance/ACC'] = acc
        return validation_loss

    def _step_callback(self, s, g, out, loss, step_idx=None) -> None:
        r"""Build and print a table summarizing the prediction of the step.

        Args:
            s (torch.Tensor)            : The network input of the step.
            g (torch.Tensor)            : The target label of the step.
            out (torch.Tensor)          : The network output.
            loss (float or torch.Tensor): The loss.
            step_idx (int, Optional)    : The number of steps.
        """
        # Print step information
        self._logger.debug(f"s: {s.shape} out: {out.shape}")
        _df, _ = self._build_validation_df(g, out)
        self._logger.debug('\n' + _df.to_string())

        # These are used for specific network and will be move to other places soon.
        if hasattr(self.net, 'module'):
            if hasattr(self.net.module, '_batch_callback'):
                self.net.module._batch_callback()
                self._logger.debug(f"LCL:{self.net.module.LCL.cutpoints}")
        elif hasattr(self.net, '_batch_callback'):
            self.net.module._batch_callback()
            self._logger.debug(f"LCL:{self.net.module.LCL.cutpoints}")