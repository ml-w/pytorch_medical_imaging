from .InferencerBase import InferencerBase
from ..med_img_dataset import DataLabel, PMIDataBase
from mnts.mnts_logger import MNTSLogger
from torch.utils.data import DataLoader
from tqdm import *
import os
import torch
import numpy as np


__all__ = ['SurvivalInferencer']

class SurvivalInferencer(InferencerBase):
    def __init__(self, in_data, out_dir, batch_size, net, checkpoint_dir, iscuda, logger=None, target_data=None,
                 config=None, **kwargs):
        inference_configs = {}
        inference_configs['indataset']      = in_data
        inference_configs['batchsize']      = batch_size
        inference_configs['net']            = net
        inference_configs['netstatedict']   = checkpoint_dir
        inference_configs['logger']         = logger
        inference_configs['outdir']         = out_dir
        inference_configs['iscuda']         = iscuda
        inference_configs['target_data']    = target_data
        inference_configs['config']         = config

        super(SurvivalInferencer, self).__init__(inference_configs)

        self._config = config
        if not self._config is None:
            self._censor_value = self._get_params_from_solver_config('censor_value', 5, True)


    def _input_check(self):
        return 0

    def _create_net(self):
        if not hasattr(self._net, 'forward'):
            in_chan = self._in_dataset[0].size()[0]
            out_chan = 2 #TODO: Temp fix

            self._logger.log_print_tqdm("Cannot create network with 'save_mask' attribute!", 20)
            self._net = self._net(in_chan, out_chan)

        self._logger.log_print_tqdm("Loading checkpoint from: " + self.net_state_dict, 20)
        self._net.load_state_dict(torch.load(self.net_state_dict), strict=False)
        # self._net = nn.DataParallel(self._net)
        self._net.train(False)
        self._net.eval()
        if self.iscuda:
            self._net = self._net.cuda()


        return self._net

    def _prepare_data(self):
        self._data_loader = DataLoader(self._in_dataset, batch_size=self.batchsize,
                                       shuffle=False, num_workers=0, drop_last=False)
        return self._data_loader


    def write_out(self):
        out_tensor = []
        last_batch_dim = 0
        with torch.no_grad():
            for index, samples in enumerate(tqdm(self._data_loader, desc="Steps")):
                # For some reason the PMIZeroBatchSampler tries to wrap the output in tuples or list this is a work
                # around that checks how many layers the input has and tries to unroll them if theres too much
                layers = 0
                _s = samples
                while isinstance(_s, list) or isinstance(_s, tuple):
                        layers += 1
                        _s = _s[0]
                if layers >= 2:
                    s = samples[0]
                else:
                    s = samples
                s = self._match_type_with_network(s)

                if isinstance(s, list):
                    self._logger.debug(f"s[0].shape: {s[0].shape}")
                    self._logger.debug(f"s[0].shape: {s[1].shape}")
                    out = self._net.forward(*s)
                else:
                    out = self._net.forward(s)

                # Check network output dimensions unsqueeze if its smaller than 2 or the dim of the last batch
                while ((out.dim() < last_batch_dim) or (out.dim()< 2)) and last_batch_dim != 0:
                    out = out.unsqueeze(0)
                    self._logger.log_print_tqdm('Unsqueezing last batch.' + str(out.shape))

                out_tensor.append(out.data.cpu())
                last_batch_dim = out.dim()
                del out, s

            out_tensor = torch.cat(out_tensor, dim=0).squeeze()
            dl = self._writter(out_tensor)
            self._logger.info('\n'+dl._data_table.to_string())

    def _writter(self, out_tensor):
        survival_table = {}
        survival_hazard = out_tensor

        # prepare output directories
        if os.path.isdir(self._outdir):
            self._outdir = os.path.join(self._outdir, 'class_inf.csv')
        if not self._outdir.endswith('.csv'):
            self._outdir += '.csv'
        if os.path.isfile(self._outdir):
            self._logger.log_print_tqdm("Overwriting file %s!"%self._outdir, 30)
        if not os.path.isdir(os.path.dirname(self._outdir)):
            os.makedirs(os.path.dirname(self._outdir), exist_ok=True)

        # prepare the output spreadsheet
        try:
            if isinstance(self._in_dataset, torch.utils.data.TensorDataset):
                survival_table['IDs'] = self._in_dataset.tensors[0].get_unique_IDs()
            elif isinstance(self._in_dataset, PMIDataBase):
                survival_table['IDs'] = self._in_dataset.get_unique_IDs()
        except:
            self._logger.exception("Could not get ID of data!")
            self._logger.warning("Falling back.")
        survival_table['Hazard'] = survival_hazard.tolist()
        dl = DataLabel.from_dict(survival_table)

        # extract survival information if it exists
        try:
            survival_gt = self._target_dataset
            if survival_gt._get_table is not None:
                dl._data_table = dl._data_table.join(survival_gt._get_table, on='IDs')
        except AttributeError or IndexError:
            self._logger.warning("Cannot access survival ground-truth. Skipping.")
        except:
            self._logger.exception("Unexpected error when trying to join with ground-truth table.")

        self._results = dl._data_table
        dl.write(self._outdir)
        return dl


    def write_out_allcps(self):
        r"""
        This load all checkpoints in the specified directory and record results and performance
        """
        import fnmatch
        import pandas as pd
        # Look for other checkpoints
        cp_dir = os.path.dirname(self.net_state_dict)
        all_cps = fnmatch.filter(os.listdir(cp_dir), "*.pt")
        all_cps.sort()

        rows = pd.DataFrame()
        for cps in tqdm(all_cps, position=1):
            self._logger.info(f"Inferencing on f{cps}")
            try:
                self._net.load_state_dict(torch.load(os.path.join(cp_dir, cps)))
            except FileNotFoundError:
                self._logger.warning(f"Cannot found file f{cps}.Skipping")
                continue
            except:
                self._logger.warning(f"Error when loading checkpoint from f{cps}. Skipping")
                continue

            # display summary look at self._results
            self._outdir = os.path.join(os.path.dirname(self._outdir), cps.replace('.pt', '.csv'))
            self.write_out()
            s = self.display_summary()
            rows = rows.append(pd.DataFrame([[cps, s]], columns=['Checkpoint', 'C-index']))
        rows.to_csv(os.path.join(os.path.dirname(self._outdir), 'AllCheckpoints.csv'))


    def display_summary(self):
        """
        This uses the C-index to measure the performance, last colume is treated as the censoring index
        """
        dl = self._results

        summary = {}
        for col in dl.columns[:-1]:
            if col == 'Hazard' or col == 'IDs':
                continue
            C = self._compute_concordance(dl['Hazard'], dl[col], (dl[col] < self._censor_value) & dl[dl.columns[-1]])
            summary[col] = C
        self._logger.info("-" * 40 + " Summary " + "-" * 40)
        for key in summary:
            self._logger.info(f"{key} - C-index: {summary[key]}")
        return summary[key]


    @staticmethod
    def _compute_concordance(risk, event_time, censor_vect):
        r"""
        Compute the concordance index. Assume no ties.

        # TODO: Handle multiple event_time classes

        .. math::

            $C-index = \frac{\sum_{i,j} I[T_j < T_i] \cdot I [\eta_j > \eta_i] d_j}{1}$

        """
        # convert everything to numpy
        risk, event_time = [np.asarray(x) for x in [risk, event_time]]

        top = bot = 0
        for i in range(len(risk)):
            # skip if censored:
            if censor_vect[i] == 0:
                continue

            times_truth = event_time > event_time[i]
            risk_truth = risk < risk[i]

            i_top = times_truth & risk_truth
            i_bot = times_truth

            top += i_top.sum()
            bot += i_bot.sum()

        c_index = top/float(bot)
        if np.isnan(c_index):
            MNTSLogger[__class__.__name__].warning("Got nan when computing concordance. Replace by 0.")
            c_index = 0

        return np.clip(c_index, 0, 1)


    def overload_dataloader(self, loader):
        self._data_loader = loader