from .ClassificationInferencer import ClassificationInferencer
from ..med_img_dataset import DataLabel
from torch.utils.data import DataLoader
from tqdm import *
import os
import torch
#from ..networks.GradCAM import *

__all__ = ['BinaryClassificationRNNInferencer']

class BinaryClassificationRNNInferencer(ClassificationInferencer):
    def __init__(self, *args, **kwargs):
        super(BinaryClassificationRNNInferencer, self).__init__(*args, **kwargs)

    def _create_net(self):
        state_dict = torch.load(self._net_state_dict, map_location=torch.device('cpu'))
        last_module = list(state_dict)[-1]

        # Read from state dict the input and output num of channels
        if not hasattr(self._net, 'forward'):
            self._logger.info("Trying to load network configs.")
            in_chan = self._in_dataset[0].size()[0] if not 'in_ch' in state_dict else \
                int(state_dict['in_ch'].item())
            out_chan = state_dict.get(last_module).shape[0] if not 'out_ch' in state_dict else \
                int(state_dict['out_ch'].item())
            self._logger.info("Creating net with in_chan: {} out_chan: {}".format(in_chan, out_chan))
            self._net = self._net(in_chan, out_chan)

        self._logger.log_print_tqdm("Loading checkpoint from: " + self._net_state_dict, 20)
        self._net.load_state_dict(state_dict, strict=False)
        # self._net = nn.DataParallel(self._net)
        self._net.train(False)
        self._net.eval()
        if self._iscuda:
            self._net = self._net.cuda()


        return self._net

    def _create_dataloader(self):
        self._data_loader = DataLoader(self._in_dataset, batch_size=self._batchsize,
                                       shuffle=False, num_workers=0, drop_last=False)
        return self._data_loader

    def write_out(self):
        out_tensor = []
        last_batch_dim = 0
        with torch.no_grad():
            for index, samples in enumerate(tqdm(self._data_loader, desc="Steps")):
                s = samples
                if (isinstance(s, tuple) or isinstance(s, list)) and len(s) > 1:
                    s = [ss.float() for ss in s]
                    if s[0].dim() < 5:
                        s[0] = s[0].unsqueeze(0)

                else:
                    s = s.float()

                if self._iscuda:
                    s = [ss.cuda() for ss in s] if isinstance(s, list) else s.cuda()

                # Squeezing output directly cause problem if the output has only one output channel.
                if isinstance(s, list):
                    out = self._net.forward(*s)
                else:
                    out = self._net.forward(s)
                if out.shape[-1] > 1:
                    out = out.squeeze()


                while ((out.dim() < last_batch_dim) or (out.dim()< 2)) and last_batch_dim != 0:
                    out = out.unsqueeze(0)
                    self._logger.log_print_tqdm('Unsqueezing last batch.' + str(out.shape))

                out_tensor.append(out.data.cpu())

                last_batch_dim = out.dim()
                del out, s

            out_tensor = torch.cat(out_tensor, dim=0) #(NxC)
            self._logger.info(f"{out_tensor}")
            dl = self._writter(out_tensor)
            self._logger.debug('\n' + dl._data_table.to_string())


    def _writter(self, out_tensor):
        out_decisions = {}
        sig_out = torch.sigmoid(out_tensor)
        out_decision = (sig_out > .5).int()
        self._num_out_out_class = int(out_tensor.shape[1]) - 1 # Because RNN add an extra stopping character.
        if os.path.isdir(self._outdir):
            self._outdir = os.path.join(self._outdir, 'class_inf.csv')
        if not self._outdir.endswith('.csv'):
            self._outdir += '.csv'
        if os.path.isfile(self._outdir):
            self._logger.log_print_tqdm("Overwriting file %s!"%self._outdir, 30)
        if not os.path.isdir(os.path.dirname(self._outdir)):
            os.makedirs(os.path.dirname(self._outdir), exist_ok=True)

        # Write decision
        out_decisions['IDs'] = self._in_dataset.get_unique_IDs()
        self._logger.debug(f"Shape: {out_tensor.shape}")
        for i in range(self._num_out_out_class):
            out_decisions[f'Prob_Class_{i}'] = sig_out[:, i].data.cpu().tolist()
            out_decisions[f'Decision_{i}'] = out_decision[:, i].tolist()
            if self._TARGET_DATASET_EXIST_FLAG:
                self._logger.debug(f"Truth of {i}")
                out_decisions[f'Truth_{i}'] = self._target_dataset._data_table.iloc[:,i].tolist()


        dl = DataLabel.from_dict(out_decisions)
        dl.write(self._outdir)
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
                             name=self._target_dataset._data_table.columns[i])
            perf = perf.append(_row)
        row = pd.Series(_get_sum_perf([TP, FP, TN, FN]), name='Overall')
        perf = perf.append(row)

        self._logger.info('\n' + perf.to_string())
        self._logger.info("Sensitivity: %.3f Specificity: %.3f NPV: %.3f PPV: %.3f OverallACC: %.3f"%(
            perf.loc['Overall']['Sensitivity'], perf.loc['Overall']['Specificity'],
            perf.loc['Overall']['NPV'], perf.loc['Overall']['PPV'], perf.loc['Overall']['ACC']
        ))
