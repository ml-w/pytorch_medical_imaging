from .ClassificationInferencer import ClassificationInferencer
from ..med_img_dataset import DataLabel
from torch.utils.data import DataLoader
from tqdm import *
from sklearn.metrics import confusion_matrix
from typing import Union, Optional, Iterable
import os
import torch
import pandas as pd
#from ..networks.GradCAM import *

__all__ = ['BinaryClassificationInferencer']


class BinaryClassificationInferencer(ClassificationInferencer):
    def __init__(self, *args, **kwargs):
        super(BinaryClassificationInferencer, self).__init__(*args, **kwargs)

    def _prepare_data(self):
        try:
            self._inference_subjects = self.pmi_data_loader._load_data_set_training(True)
        except:
            self._inference_subjects = self.pmi_data_loader._load_data_set_inference()

    def _reshape_tensors(self,
                         out_list: Iterable[torch.FloatTensor],
                         gt_list: Iterable[torch.FloatTensor]):
        r"""Align shape before putting them into _writter

        Args:
            out_list:
                List of tensors with dimension (1 x C)
            gt_list:
                List of tensors with either dimensino (1 x 1) or (1)

        Returns:
            out_tensor: (B x C)
            gt_list: (B x 1)
        """
        out_tensor = torch.cat(out_list, dim=0) #(NxC)
        gt_tensor = torch.cat(gt_list, dim=0).reshape_as(out_tensor) if len(gt_list) > 0 else None
        return out_tensor, gt_tensor

    def _writter(self,
                 out_tensor: torch.IntTensor,
                 uids: Iterable[Union[str, int]],
                 gt: Optional[torch.IntTensor] = None,
                 sig_out=True):
        r"""Convert the output into a table

        Args:
            out_tensor (torch.IntTensor):
                Tensor with dimension (B x C) where C is the number of classes.
            uids (iterable):
                Iterable with the same length as `out_tensor`.
            gt (Optional, torch.IntTensor):
                Tensor with dimension (B x 1).
            sig_out (Optional, bool):
                If the output is required to go through the sigmoid function.

        Returns:
            out: DataLabel
        """
        out_decisions = {}
        out_tensor = torch.sigmoid(out_tensor) if sig_out else out_tensor
        out_decision = (out_tensor > .5).int()
        self._num_out_out_class = int(out_tensor.shape[1])
        if os.path.isdir(self.output_dir):
            self.outdir = os.path.join(self.output_dir, 'class_inf.csv')
        if not self.outdir.endswith('.csv'):
            self.outdir += '.csv'
        if os.path.isfile(self.outdir):
            self._logger.log_print_tqdm("Overwriting file %s!"%self.outdir, 30)
        if not os.path.isdir(os.path.dirname(self.outdir)):
            os.makedirs(os.path.dirname(self.outdir), exist_ok=True)

        # Write decision
        out_decisions['IDs'] = uids
        for i in range(out_tensor.shape[1]):
            out_decisions[f'Prob_Class_{i}'] = out_tensor[:, i].data.cpu().tolist()
            out_decisions[f'Decision_{i}'] = out_decision[:, i].tolist()
            if gt is not None:
                # if gt is one single column vector with same shape as out_decision
                self._logger.error(f"fucking gt: {gt}")
                if gt.shape[1] == out_tensor.shape[1]:
                    out_decisions[f'Truth_{i}'] = gt[:, i].tolist()
                else:
                    out_decisions[f'Truth_{i}'] = gt.flatten().tolist()
                self._TARGET_DATASET_EXIST_FLAG = True
            else:
                self._TARGET_DATASET_EXIST_FLAG = False

        import pprint
        self._logger.debug(f"out_decision: {pprint.pformat(out_decisions)}")
        dl = DataLabel.from_dict(out_decisions)
        dl.write(self.outdir)
        self._dl = dl
        return dl

    def display_summary(self):
        """
        Called if target_dir is provided in the config file.
        Display the sensitivity, specificity, NPV, PPV and accuracy of the classification.
        """
        import pandas as pd


        if not hasattr(self, '_dl'):
            self._logger.log_print_tqdm("Cannot find data. Have you called _writter() yet?", 30)
            return

        if not self._TARGET_DATASET_EXIST_FLAG:
            self._logger.log_print_tqdm("No target data provided. No summary to display.", 20)
            return

        subdf = self._dl._data_table.copy()
        for i in range(self._num_out_out_class):
            _subdf = subdf[['%s_%s'%(a, i) for a in ['Prob_Class', 'Decision', 'Truth']]]
            subdf['perf_%s'%i] = _subdf[[f'Decision_{i}', f'Truth_{i}']].apply(BinaryClassificationInferencer._get_perf, axis=1)

        # compute sensitivity, specificity ...etc
        perf = pd.DataFrame()
        TP, TN, FP, FN = [0, 0, 0, 0]
        for i in range(self._num_out_out_class):
            _col = subdf[f'perf_{i}']
            _TP = (_col == 'TP').sum()
            _TN = (_col == 'TN').sum()
            _FP = (_col == 'FP').sum()
            _FN = (_col == 'FN').sum()
            TP += _TP
            TN += _TN
            FP += _FP
            FN += _FN
            _row = pd.Series(BinaryClassificationInferencer._get_sum_perf([_TP, _FP, _TN, _FN]),
                             name=f"Class {i}")
            perf = perf.append(_row)
        row = pd.Series(BinaryClassificationInferencer._get_sum_perf([TP, FP, TN, FN]), name='Overall')
        perf = perf.append(row)

        self._logger.info('\n' + perf.to_string())
        self._logger.info("Sensitivity: %.3f Specificity: %.3f NPV: %.3f PPV: %.3f OverallACC: %.3f"%(
            perf.loc['Overall']['Sensitivity'], perf.loc['Overall']['Specificity'],
            perf.loc['Overall']['NPV'], perf.loc['Overall']['PPV'], perf.loc['Overall']['ACC']
        ))

    @staticmethod
    def _get_perf(s):
        predict, truth = s
        if truth:
            if predict == truth:
                return 'TP'
            else:
                return 'FN'
        else:
            if predict == truth:
                return 'TN'
            else:
                return 'FP'

    @staticmethod
    def _get_sum_perf(perf_counts):
        TP, FP, TN, FN = [float(a) for a in perf_counts]
        sens    = TP / (TP + FN + 1E-16)
        spec    = TN / (TN + FP + 1E-16)
        npv     = TN / (TN + FN + 1E-16)
        ppv     = TP / (TP + FP + 1E-16)
        acc     = (TP + TN) / (TP + TN + FP + FN)
        return {'Sensitivity': sens, 'Specificity': spec, 'NPV': npv, 'PPV': ppv, 'ACC': acc}