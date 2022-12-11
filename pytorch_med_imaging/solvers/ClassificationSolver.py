from .SolverBase import SolverBase
from mnts.mnts_logger import MNTSLogger


from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np
from tqdm import *
from ..loss import CumulativeLinkLoss

import pandas as pd

__all__ = ['ClassificationSolver']

class ClassificationSolver(SolverBase):
    def __init__(self, net, hyperparam_dict, use_cuda):
        r"""Solver for classification tasks. For details to kwargs, see :class:`SolverBase`.

        Attributes:
            sigmoid_params (dict, Optional):
                This controls the change in weights between background (0) and non-background during loss
                evaluation.
            class_weights (list, Optional):
                If specified, this will be the fixed class weights that will be passed to the loss function.
                This is ignored if ordinal_class is specified to True. Default to None.
            ordinal_class (bool, Optional):
                If True, the ground-truth is expected to be ordinal class starting from 0. The BCE with logit loss
                will be used and the ground-truth will be encoded. See :func:`_pred2label4ordinal()` for more.
                Default to False
        """
        super(ClassificationSolver, self).__init__(net, hyperparam_dict, use_cuda)

    def _load_config(self, default_attr):
        r"""Inherit this to get more default hyperparameters"""
        _default_attr = {
            'solverparams_sigmoid_params'   : {'delay': 15, 'stretch': 2, 'cap': 0.3},
            'solverparams_class_weights'    : None,
            'solverparams_decay_init_weight': 0,
            'solverparams_ordinal_class'    : False,
            'solverparams_ordinal_mse'      : False
        }
        if isinstance(default_attr, dict):
            _default_attr.update(default_attr)
        super(ClassificationSolver, self)._load_config(_default_attr)

    def create_lossfunction(self):
        # set class weights to 0 to disable class weight for loss function
        try:
            if not self.solverparams_class_weights == 0:
                if not isinstance(self.solverparams_class_weights, (list, tuple, torch.Tensor)) and \
                        self.solverparams_class_weights is not None:
                    self.solverparams_class_weights = [self.solverparams_class_weights]
                self.solverparams_class_weights = torch.as_tensor(self.solverparams_class_weights)
                loss_init_weights = self.solverparams_class_weights.cpu().float()
                self._logger.info("Initial weight factor: " + str(self.solverparams_class_weights))
            else:
                self._logger.info("Skipping class weights.")
                loss_init_weights = None
        except Exception as e:
            self._logger.warning("Weight convertion to tensor fails. Falling back!")
            self._logger.debug(f"Original error: {e}")
            loss_init_weights = None

        if self.solverparams_ordinal_class:
            self._logger.info("Using ordinal classification mode.")
            self.lossfunction = CumulativeLinkLoss(class_weights=loss_init_weights)
        elif self.solverparams_ordinal_mse:
            self._logger.info("Using ordinal mse mode.")
            self.lossfunction = nn.SmoothL1Loss()
        else:
            self.lossfunction = nn.CrossEntropyLoss(weight=loss_init_weights) #TODO: Allow custom loss function

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

        self._logger.debug(f"s: {s.shape} out: {out.shape}")
        # Print step information
        _df, _ = self._build_validation_df(g, out)
        self._logger.debug('\n' + _df.to_string())
        del _df
        return out

    def _build_validation_df(self, g, res, uid=None):
        _df = pd.DataFrame.from_dict({f'res_{d}': list(res[:, d].cpu().detach().numpy())
                                      for d in range(res.shape[-1])})
        if not self.solverparams_ordinal_mse:
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
        out, s, g = args

        if self.iscuda:
            s = self._match_type_with_network(s)
            g = self._match_type_with_network(g)

        out = out.squeeze() # Expect (B x C) where C is same as number of classes
        if g.squeeze().dim() == 1 and not self.solverparams_ordinal_mse:
            g = g.squeeze().long()
        self._logger.debug(f"Output size out: {out.shape} g: {g.shape}")
        # Cross entropy does not need any processing, just give the raw output
        loss = self.lossfunction(out, g)
        return loss

    def validation(self):
        if self.data_loader_val is None:
            self._logger.warning("Validation skipped because no loader is available.")
            return None
        with torch.no_grad():
            self.net.eval()

            decisions = []
            validation_loss = []
            for mb in tqdm(self.data_loader_val, desc="Validation", position=2):
                s, g = self._unpack_minibatch(mb, self.solverparams_unpack_keys_forward)
                s = self._match_type_with_network(s)
                g = self._match_type_with_network(g)

                if isinstance(s, list):
                    res = self.net(*s)
                else:
                    res = self.net(s)
                # res = torch.(res, dim=1)
                while res.dim() < 2:
                    res = res.unsqueeze(0)

                if not self.solverparams_ordinal_mse:
                    dic = torch.argmax(torch.softmax(res, dim=1), dim=1)
                else:
                    dic = torch.round(res).long()
                decisions.extend([guess == truth for guess, truth in zip(dic.tolist(), g.tolist())])
                loss = self._loss_eval(res, s, g)
                validation_loss.append(loss.item())

            # Compute accuracies
            acc = float(decisions.count(True)) / float(len(decisions))
            validation_loss = np.mean(np.array(validation_loss).flatten())
            self._logger.log_print_tqdm("Validation Result - ACC: %.05f, VAL: %.05f"%(acc, validation_loss))

        self.plotter_dict['scalars']['Loss/Validation Loss'] = validation_loss
        self.plotter_dict['scalars']['Performance/ACC'] = acc
        return validation_loss

    def _step_callback(self, s, g, out, loss, step_idx=None):
        if hasattr(self.net, 'module'):
            if hasattr(self.net.module, '_batch_callback'):
                self.net.module._batch_callback()
                self._logger.debug(f"LCL:{self.net.module.LCL.cutpoints}")
        elif hasattr(self.net, '_batch_callback'):
            self.net.module._batch_callback()
            self._logger.debug(f"LCL:{self.net.module.LCL.cutpoints}")