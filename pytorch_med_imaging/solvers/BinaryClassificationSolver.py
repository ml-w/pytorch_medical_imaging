from .ClassificationSolver import ClassificationSolver
from mnts.mnts_logger import MNTSLogger

from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
from ..loss import BinaryFocalLoss, TverskyDiceLoss
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pprint
from tqdm import tqdm
from typing import Union, Iterable, Any, Tuple

__all__ = ['BinaryClassificationSolver']


class BinaryClassificationSolver(ClassificationSolver):
    r"""Solver for binary classification tasks. Basically this is the same as :class:`ClassificationSolver`. It
    just edit the validation and performance calculation to better monitor the training progress. In addition,
    it will also align the dimension and size of the tensor. In addition, this class also enables binary prediction
    of multiple questions.  Typically, the network output should have a dimension :math:`(B × C)` where :math:`C`
    is the number of question to answer. The target label should have the same dimension as the network output.

    Args:
        cfg (ClassificationSolverCFG):
            The configuration file.
    """
    def __init__(self,
                 cfg: ClassificationSolver,
                 *args, **kwargs):
        super(ClassificationSolver, self).__init__(cfg, *args, **kwargs)

        self._validation_misclassification_record = {}

    def _validation_step_callback(self, g: torch.Tensor, res: torch.Tensor, loss: Union[torch.Tensor, float],
                                  uids=None) -> None:
        r"""Uses :attr:`perf` to store the dictionary of various data."""
        self.validation_losses.append(loss.item())
        if len(self.perfs) == 0:
            self.perfs.append({
                'dics'       : [],
                'gts'        : [],
                'predictions': [],
                'uids'       : []
            })
        store_dict = self.perfs[0]
        g, res = self._align_g_res_size(g, res)
        _df, dic = self._build_validation_df(g, res)

        # Decision were made by checking whether value is > 0.5 after sigmoid
        store_dict['dics'].extend(dic)
        store_dict['gts'].extend(g)
        store_dict['predictions'].extend(res.flatten().tolist())
        if isinstance(uids, (tuple, list)):
            store_dict['uids'].extend(uids)

    def _validation_callback(self) -> None:
        r"""Compute performance and put the performance into tensorboard for plotting."""
        store_dict  = self.perfs[0]
        dics        = store_dict['dics']
        gts         = store_dict['gts']
        predictions = store_dict['predictions']
        uids        = store_dict['uids']

        acc, per_mean, res_table = self._compute_performance(dics, gts)
        validation_loss = np.mean(np.array(self.validation_losses).flatten())
        df = {'prediction': predictions,
              'decision'  : torch.cat(dics, dim = 0).bool().tolist(),
              'truth'     : torch.cat(gts , dim = 0).bool().tolist()}
        df = pd.DataFrame(data=df, index=uids)
        df['correct'] = df['decision'] == df['truth']
        self._logger.info(f"\n{df.to_string()}")
        # self._logger.debug("_val_perfs: \n%s"%res_table.T.to_string())
        self._logger.info("Validation Result - ACC: %.05f, VAL: %.05f"%(acc, validation_loss))
        self.plotter_dict['scalars']['val/loss'] = validation_loss
        self.plotter_dict['scalars']['val/performance/ACC'] = acc
        for param, val in per_mean.items():
            self.plotter_dict['scalars']['val/performance/%s'%param] = val

        # Print the misclassification report
        if len(self._validation_misclassification_record) > 0:
            self._logger.info("Validation misclassification report: {}".format(
                pprint.pformat(self._validation_misclassification_record)
            ))


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
            Tuple[float, pd.Series, pd.DataFrame]: Accuracy value. The mean performance. The summary of the performance.
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

    def _align_g_res_size(self,
                          g: torch.Tensor,
                          res: torch.Tensor):
        r"""For ordinary classification, expects network output `res` and target labels `g` dimension to be
        :math:`(B × C)` and :math:`(B × 1)` where :math:`C` is the number of classes in label. For binary classification
        here, we allow users to ask more than one binary questions such that the network output and the labels should
        both have a dimension of :math:`(B × C)` where :math:`C` is the number of questions asked.

        Generally speaking, it is common to have :math:`C = 1`, but the dimension 1 poses trouble because it gets
        squeezed by calling `torch.Tensor.squeeze`.

        Args:
            g (torch.Tensor):
                The target label tensor. Should have a dimension of :math:`(B × C)`, but if not, it gets reshaped.
            res (torch.Tensor):
                The network output tensor. Should have a dimension of :math:`(B × C)`.

        Returns:
            torch.Tensor: The reshaped target label. The reshaped output tensor.

        """
        self._logger.debug(f"Before align: res_size = {res.shape}; g_size {g.shape}")
        if not res.dim() == 2:
            msg = f"Network output should have a dimension (B × C), but got: {list(res.shape)} instead."
            raise IndexError(msg)
        num_q = res.shape[1]
        while g.dim() < 2:
            g = g.unsqueeze(-1)
        if not g.shape[1] == num_q:
            g = g.reshape(-1, num_q)
        self._logger.debug(f"After align: res_size = {res.shape}; g_size = {g.shape}")
        return g, res

    def _loss_eval(self, *args):
        r""""""
        out, s, g = args
        g, out = self._align_g_res_size(g, out)

        # ordinal_mse and ordinal_class is not available
        if self.ordinal_mse or self.ordinal_class:
            self._logger.warning("Original classification is not enabled for binary predictions.")
            self.ordinal_mse = False
            self.ordinal_class = False
        return super(BinaryClassificationSolver, self)._loss_eval(out, s, g)

