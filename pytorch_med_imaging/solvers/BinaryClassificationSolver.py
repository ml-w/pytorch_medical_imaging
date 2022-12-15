from .ClassificationSolver import ClassificationSolver
from mnts.mnts_logger import MNTSLogger

from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
from ..loss import FocalLoss, TverskyDiceLoss
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, Iterable, Any, Tuple

__all__ = ['BinaryClassificationSolver']


class BinaryClassificationSolver(ClassificationSolver):
    def __init__(self,
                 net: torch.nn.modules,
                 hyperparam_dict: dict,
                 use_cuda: bool):
        """Solver for binary classification tasks. For details to kwargs, see :class:`SolverBase`.
        """
        super(ClassificationSolver, self).__init__(net, hyperparam_dict, use_cuda)

    def prepare_lossfunction(self):
        super(BinaryClassificationSolver, self).prepare_lossfunction()
        # Simple error check
        if not isinstance(self.solverparams_class_weights, (float, torch.FloatTensor)) \
                and self.solverparams_class_weights is not None:
            msg = f"Class weight for binary classifier must be a single float number. Got " \
                  f"{self.solverparams_class_weights} instead."
            raise AttributeError(msg)

        try:
            self.lossfunction = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=self.solverparams_class_weights) # Combined with sigmoid.
            self._logger.info(f"Using BCEWithLogitLoss, pos_weight = {self.solverparams_class_weights}")
        except:
            self.lossfunction = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=self.lossfunction.weight) # Combined with sigmoid.
            self._logger.info(f"Using BCEWithLogitLoss, pos_weight = {self.lossfunction.weight}")

    def validation(self):
        if self.data_loader_val is None:
            self._logger.warning("Validation skipped because no loader is available.")
            return None

        with torch.no_grad():
            self.net = self.net.eval()

            validation_loss = []
            dics = []
            gts = []
            predictions = []
            uids = []

            for mb in tqdm(self.data_loader_val, desc="Validation", position=2):
                # s: (B x num_class), g: (B x 1)GGq
                s, g = self._unpack_minibatch(mb, self.solverparams_unpack_keys_forward)
                s = self._match_type_with_network(s)
                g = self._match_type_with_network(g)

                try:
                    self._logger.debug(f"Before call s_size = {s.shape}; g_size = {g.shape}")
                except:
                    pass

                if isinstance(s, list):
                    res = self.net(*s)
                else:
                    res = self.net(s)

                # align dimensions
                while res.dim() < 2:
                    res = res.unsqueeze(0)
                g = self._align_g_res_size(g, res)
                self._logger.debug(f"After call g_size = {g.shape}, res_size = {res.shape}")

                # No sigmoid function
                loss = self._loss_eval(res, s, g)
                _df, dic = self._build_validation_df(g, res)
                self._logger.debug(f"_loss: {loss}")
                self._logger.debug("_val_res:\n" + _df.to_string())
                self._logger.debug("_val_step_loss: {}".format(loss.data.item()))
                # Decision were made by checking whether value is > 0.5 after sigmoid
                dics.extend(dic.cpu())
                gts.extend(g.cpu())
                predictions.extend(res.cpu().flatten().tolist())
                uids.extend(mb['uid'])
                validation_loss.append(loss.item())
                # tqdm.write(str(torch.stack([torch.stack([a, b, c]) for a, b, c, in zip(dic, torch.sigmoid(res), g)])))
                del dic, s, g, _df

        acc, per_mean, res_table = self._compute_performance(dics, gts)
        validation_loss = np.mean(np.array(validation_loss).flatten())
        df = {'prediction': predictions,
              'decision': torch.cat(dics, dim=0).bool().tolist(),
              'truth': torch.cat(gts, dim=0).bool().tolist()}
        df = pd.DataFrame(data=df, index=uids)
        df['correct'] = df['decision'] == df['truth']
        self._logger.info(f"\n{df.to_string()}")
        # self._logger.debug("_val_perfs: \n%s"%res_table.T.to_string())
        self._logger.info("Validation Result - ACC: %.05f, VAL: %.05f"%(acc, validation_loss))
        self.net = self.net.train()
        self.plotter_dict['scalars']['Loss/Validation Loss'] = validation_loss
        self.plotter_dict['scalars']['Performance/ACC'] = acc
        for param, val in per_mean.iteritems():
            self.plotter_dict['scalars']['Performance/%s'%param] = val
        self.optimizer.zero_grad()
        return validation_loss

    @staticmethod
    def _compute_performance(dics: Iterable[torch.IntTensor],
                             gts: Iterable[torch.IntTensor]) -> Tuple[float, pd.Series, pd.DataFrame]:
        r"""Compute the performance as a table in terms of accuracy, sensitivity, specificity,
        positive and negative predictive values.

        Args:
            dics (torch.IntTensor or list):
                This should be a list of decisions computed from the network output.
            gt (torch.IntTensor or list):
                This should be the ground-truths.

        Returns:
            acc (pd.float):
                Accuracy value.
            per_mean (pd.DataFrame or pd.Series):
                The mean performance.
            res_table (pd.DataFrame):
                The summary of the performance.
        """
        # Compute accuracies
        # Stack the decisions per batch first
        dics = torch.cat(dics, dim=0).bool()
        gts = torch.cat(gts, dim=0).bool()
        tn, fp, fn, tp = confusion_matrix(gts.numpy().ravel(), dics.numpy().ravel(), labels=[0, 1]).ravel()

        accuracy = pd.Series((tp + tn) / float(tp + tn + fp + fn), name='Accuracy')
        sens = pd.Series(tp / (1E-32 + float(tp + fn)), name = 'Sensitivity')
        spec = pd.Series(tn / (1E-32 + float(tn + fp)), name = 'Specificity')
        ppv  = pd.Series(tp / (1E-32 + float(tp + fp)), name = 'PPV')
        npv  = pd.Series(tn / (1E-32 + float(tn + fn)), name = 'NPV')
        res_table = pd.concat([accuracy, sens, spec, ppv, npv], axis=1)
        per_mean = res_table.mean()
        acc = accuracy.mean()
        return acc, per_mean, res_table

    def _build_validation_df(self, g, res, uid=None):
        _pairs = zip(res.flatten().data.cpu().numpy(),
                     g.flatten().data.cpu().numpy(),
                     torch.sigmoid(res).flatten().data.cpu().numpy())
        _df = pd.DataFrame(_pairs, columns=['res', 'g', 'sig_res'])
        if uid is not None:
            _df.index = uid

        # model_output: (B x num_class)
        dic = torch.zeros_like(res)
        pos = torch.where(torch.sigmoid(res) > 0.5)
        dic[pos] = 1
        return _df, dic

    def _align_g_res_size(self, g, res):
        r"""Work arround, normally we don't need this if we can shape ground-truth correctly. For classification
        this should always be (B x C) where C is number of classes. Assume for binary classification, C = 1"""
        self._logger.debug(f"Before align: res_size = {res.shape}; g_size {g.shape}")
        g = g.view(-1, 1)
        self._logger.debug(f"After align: res_size = {res.shape}; g_size = {g.shape}")
        return g


    def step(self, *args):
        s, g = args
        try:
            self._logger.debug(f"step(): s_size = {s.shape};g_size = {g.shape}")
        except:
            pass

        # Skip if all ground-truth have the same type
        # if g.unique().shape[0] == 1:
        #     with torch.no_grad():
        #         out = self._feed_forward(*args)
        #         loss = self._loss_eval(out, *args)
        #         # loss.backward()
        #         # Cope with extreme data imbalance.
        #         self._logger.warning("Skipping grad, all input are the same class.")
        #         self._called_time += 1
        #     return out, loss.cpu().data
        # else:
        out = self._feed_forward(*args)
        loss = self._loss_eval(out, *args)
        self._update_network(loss)

        # if schedular is OneCycleLR
        if isinstance(self.lr_sche, lr_scheduler.OneCycleLR):
            self.lr_sche.step()
        self._step_called_time += 1
        return out, loss.cpu().data

    def _loss_eval(self, *args):
        out, s, g = args
        #out (B x C) g (B x C)
        out = self._match_type_with_network(out)
        g = self._match_type_with_network(g)

        if out.shape[0] > 1:
            out = out.squeeze().unsqueeze(1)
            g = g.squeeze().unsqueeze(1)
        self._logger.debug(f"_loss_eval size - out: {out.shape} g: {g.shape}")

        # An issues is caused if the batchsize is 1, this is a work arround.
        if out.shape[0] == 1:
            loss = self.lossfunction(out.squeeze().unsqueeze(0), g.squeeze().unsqueeze(0))
        else:
            loss = self.lossfunction(out, g)
        return loss

