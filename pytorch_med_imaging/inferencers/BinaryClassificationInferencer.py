from .ClassificationInferencer import ClassificationInferencer
from ..med_img_dataset import DataLabel
from torch.utils.data import DataLoader
from tqdm import *
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

    def write_out(self):
        uids = []
        gt = []
        out_tensor = []
        last_batch_dim = 0
        with torch.no_grad():
            dataloader = DataLoader(self._inference_subjects, batch_size=self.batch_size, shuffle=False)
            for index, mb in enumerate(tqdm(dataloader, desc="Steps")):
                s = self._unpack_minibatch(mb, self.unpack_keys_inf)
                s = self._match_type_with_network(s)

                self._logger.debug(f"s size: {s.shape if not isinstance(s, list) else [ss.shape for ss in s]}")

                # Squeezing output directly cause problem if the output has only one output channel.
                if isinstance(s, list):
                    out = self.net.forward(*s)
                else:
                    out = self.net.forward(s)
                if out.shape[-1] > 1:
                    out = out.squeeze()

                while ((out.dim() < last_batch_dim) or (out.dim() < 2)) and last_batch_dim != 0:
                    out = out.unsqueeze(0)
                    self._logger.log_print_tqdm('Unsqueezing last batch.' + str(out.shape))

                out_tensor.append(out.data.cpu())
                uids.extend(mb['uid'])
                if 'gt' in mb:
                    gt.append(mb['gt'])

                last_batch_dim = out.dim()
                del out, s

            out_tensor = torch.cat(out_tensor, dim=0) #(NxC)
            gt_tensor = torch.cat(gt, dim=0).reshape_as(out_tensor)
            dl = self._writter(out_tensor, uids, gt_tensor)
            self._logger.debug('\n' + dl._data_table.to_string())


    def _writter(self, out_tensor, uids, gt = None):
        out_decisions = {}
        sig_out = torch.sigmoid(out_tensor)
        out_decision = (sig_out > .5).int()
        self._num_out_out_class = int(out_tensor.shape[1])
        if os.path.isdir(self.outdir):
            self.outdir = os.path.join(self.outdir, 'class_inf.csv')
        if not self.outdir.endswith('.csv'):
            self.outdir += '.csv'
        if os.path.isfile(self.outdir):
            self._logger.log_print_tqdm("Overwriting file %s!"%self.outdir, 30)
        if not os.path.isdir(os.path.dirname(self.outdir)):
            os.makedirs(os.path.dirname(self.outdir), exist_ok=True)

        # Write decision
        out_decisions['IDs'] = uids
        for i in range(out_tensor.shape[1]):
            out_decisions[f'Prob_Class_{i}'] = sig_out[:, i].data.cpu().tolist()
            out_decisions[f'Decision_{i}'] = out_decision[:, i].tolist()
            if gt is not None:
                out_decisions[f'Truth_{i}'] = gt[:, i].tolist()
                self._TARGET_DATASET_EXIST_FLAG = True

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

        def _get_sum_perf(perf_counts):
            TP, FP, TN, FN = [float(a) for a in perf_counts]
            sens    = TP / (TP + FN + 1E-8)
            spec    = TN / (TN + FP + 1E-8)
            npv     = TN / (TN + FN + 1E-8)
            ppv     = TP / (TP + FP + 1E-8)
            acc     = (TP + TN) / (TP + TN + FP + FN)
            return {'Sensitivity': sens, 'Specificity': spec, 'NPV': npv, 'PPV': ppv, 'ACC': acc}

        if not hasattr(self, '_dl'):
            self._logger.log_print_tqdm("Cannot find data. Have you called _writter() yet?", 30)
            return

        if not self._TARGET_DATASET_EXIST_FLAG:
            self._logger.log_print_tqdm("No target data provided. No summary to display.", 20)
            return

        subdf = self._dl._data_table.copy()
        for i in range(self._num_out_out_class):
            _subdf = subdf[['%s_%s'%(a, i) for a in ['Prob_Class', 'Decision', 'Truth']]]
            subdf['perf_%s'%i] = _subdf[[f'Decision_{i}', f'Truth_{i}']].apply(_get_perf, axis=1)

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
            _row = pd.Series(_get_sum_perf([_TP, _FP, _TN, _FN]),
                             name=self._dl._data_table.columns[i])
            perf = perf.append(_row)
        row = pd.Series(_get_sum_perf([TP, FP, TN, FN]), name='Overall')
        perf = perf.append(row)

        self._logger.info('\n' + perf.to_string())
        self._logger.info("Sensitivity: %.3f Specificity: %.3f NPV: %.3f PPV: %.3f OverallACC: %.3f"%(
            perf.loc['Overall']['Sensitivity'], perf.loc['Overall']['Specificity'],
            perf.loc['Overall']['NPV'], perf.loc['Overall']['PPV'], perf.loc['Overall']['ACC']
        ))
