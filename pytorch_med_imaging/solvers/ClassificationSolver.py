from .SolverBase import SolverBase
from mnts.mnts_logger import MNTSLogger


from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np
from tqdm import *

import pandas as pd

__all__ = ['ClassificationSolver']

class ClassificationSolver(SolverBase):
    def __init__(self, net, hyperparam_dict, use_cuda):
        r"""Solver for classification tasks. For details to kwargs, see :class:`SolverBase`.
        """
        super(ClassificationSolver, self).__init__(net, hyperparam_dict, use_cuda)

    def _load_default_attr(self, default_attr):
        r"""Inherit this to get more default hyperparameters"""
        _default_attr = {
            'solverparams_sigmoid_params'   : {'delay': 15, 'stretch': 2, 'cap': 0.3},
            'solverparams_class_weights'    : None,
            'solverparams_decay_init_weight': 0
        }
        if isinstance(default_attr, dict):
            _default_attr.update(default_attr)
        super(ClassificationSolver, self)._load_default_attr(_default_attr)

    def create_lossfunction(self):
        # set class weights to 0 to disable class weight for loss function
        try:
            if not self.solverparams_class_weights == 0:
                weights = torch.as_tensor(self.solverparams_class_weights)
                loss_init_weights = weights.cpu().float()
                self._logger.info("Initial weight factor: " + str(weights))
            else:
                self._logger.info("Skipping class weights.")
                loss_init_weights = None
        except Exception as e:
            self._logger.warning("Weight convertion to tensor fails. Falling back!")
            loss_init_weights = None
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

        # Print step information
        _df = pd.DataFrame.from_dict({f'res_{d}': list(out[:,d].cpu().detach().numpy())
                                      for d in range(out.shape[-1])})
        _df_gt = pd.DataFrame.from_dict({'gt': list(g.flatten().cpu().detach().numpy())})
        _df_sigres = pd.DataFrame.from_dict({f'sig_{d}': list(torch.sigmoid(out[:,d]).cpu().detach().numpy())
                                             for d in range(out.shape[-1])})
        _df = pd.concat([_df, _df_gt, _df_sigres], axis=1)
        self._logger.debug('\n' + _df.to_string())
        del _df
        return out

    def _loss_eval(self, *args):
        out, s, g = args
        if self.iscuda:
            g = self._force_cuda(g)

        out = out.squeeze()
        g = g.squeeze().long()
        self._logger.debug(f"Output size out: {out.shape} g: {g.shape}")
        loss = self.lossfunction(out, g)
        return loss

    def validation(self):
        if self._data_loader_val is None:
            self._logger.warning("Validation skipped because no loader is available.")
            return None
        with torch.no_grad():
            self.net.eval()

            decisions = []
            validation_loss = []
            for mb in tqdm(self._data_loader_val, desc="Validation", position=2):
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
        return validation_loss

