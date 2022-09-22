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

    def _load_default_attr(self, default_attr):
        r"""Inherit this to get more default hyperparameters"""
        _default_attr = {
            'solverparams_sigmoid_params'   : {'delay': 15, 'stretch': 2, 'cap': 0.3},
            'solverparams_class_weights'    : None,
            'solverparams_decay_init_weight': 0,
            'solverparams_ordinal_class'    : False,
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

        if not self.solverparams_ordinal_class:
            self.lossfunction = nn.CrossEntropyLoss(weight=loss_init_weights) #TODO: Allow custom loss function
        else:
            self.lossfunction = nn.BCEWithLogitsLoss()

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
        _df = pd.concat([_df, _df_gt], axis=1)
        if self.solverparams_ordinal_class:
            _df['Predicted'] = self._pred2label4ordinal(torch.sigmoid(out.squeeze())).cpu().detach().numpy()
        else:
            _df['Predicted'] = torch.argmax(out.squeeze(), dim=1).cpu().detach().numpy()
        self._logger.debug('\n' + _df.to_string())
        del _df
        return out

    def _loss_eval(self, *args):
        out, s, g = args

        if self.iscuda:
            s = self._match_type_with_network(s)
            g = self._match_type_with_network(g)

        out = out.squeeze() # Expect (B x C) where C is same as number of classes
        g = g.squeeze().long()
        self._logger.debug(f"Output size out: {out.shape} g: {g.shape}")
        if self.solverparams_ordinal_class:
            # g expected to be long labels with a single dimension where max value label equal to # output channels (B)
            # encode g such that, for cases with max number of class = 5
            #   0 = [0, 0, 0, 0, 0]
            #   1 = [1, 0, 0, 0, 0]
            #   2 = [1, 1, 0, 0, 0]
            #   3 = [1, 1, 1, 0, 0]
            #   4 = [1, 1, 1, 1, 0]
            #   5 = [1, 1, 1, 1, 1]
            num_batch = out.shape[0]
            num_chan = out.shape[1]
            new_g = torch.zeros([num_batch, num_chan], dtype=torch.long)
            for i in range(num_batch):
                new_g[i, 0:g[i]+1] = 1
            new_g = self._match_type_with_network(new_g)
            loss = self.lossfunction(out, new_g)
        else:
            # Cross entropy does not need any processing, just give the raw output
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

                if self.solverparams_ordinal_class:
                    dic = self._pred2label4ordinal(torch.sigmoid(res))
                else:
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

    def _build_validation_df(self, g, res):
        chan = res.shape[-1] # chan should be equal to number of classes
        _data =np.concatenate([res.view(-1, chan).data.cpu().numpy(), g.data.cpu().numpy()], axis=-1)
        _df = pd.DataFrame(data=_data, columns=['res_%i'%i for i in range(chan)] + ['g'])
        _df['Predicted'] = torch.argmax(res, dim=1).long().cpu()

    @staticmethod
    def _pred2label4ordinal(pred: torch.FloatTensor):
        r"""Convert encoded predictions back to class
            0 <- [0.1, 0.1, 0.1, 0.1, 0.1]
            1 <- [0.9, 0.1, 0.1, 0.1, 0.1]
            2 <- [0.8, 0.9, 0.1, 0.2, 0.1]
            3 <- [0.6, 0.9, 0.7, 0.2, 0.1]
            so on ...

        """
        return (pred > 0.5).cumprod(dim=1).sum(dim=1) - 1