from .InferencerBase import InferencerBase
from ..med_img_dataset import ImageDataSet, DataLabel, ImageDataMultiChannel
from torch.utils.data import DataLoader
from tqdm import *
from torch.autograd import Variable
import os
import torch
import torch.nn.functional as F
import numpy as np
from SimpleITK import WriteImage, ReadImage, GetImageFromArray
from ..networks.GradCAM import *
from torchvision.utils import make_grid
from imageio import imsave
from pytorch_med_imaging.Algorithms.visualization import draw_overlay_heatmap

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

        self._logger.log_print_tqdm("Loading checkpoint from: " + self._net_state_dict, 20)
        self._net.load_state_dict(torch.load(self._net_state_dict), strict=False)
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
        survival_harzard = out_tensor

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
            survival_table['IDs'] = self._in_dataset.tensors[0].get_unique_IDs()
        except:
            self._logger.error("Could not get ID of data!")
        survival_table['Harzard'] = survival_harzard.tolist()
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

    def display_summary(self):
        """
        This uses the C-index to measure the performance
        """
        dl = self._results

        summary = {}
        for col in dl.columns:
            if col == 'Harzard' or col == 'IDs':
                continue
            C = self._compute_concordance(dl['Harzard'], dl[col], self._censor_value)
            summary[col] = C
        self._logger.info(f"\n{summary}")


    @staticmethod
    def _compute_concordance(risk, event_time, censor_thres):
        r"""
        Compute the concordance index.

        .. math::

            $C-index = \frac{\sum_{i,j} I[T_j < T_i] \cdot I [\eta_j > \eta_i] d_j}{1}$

        """
        # convert everything to numpy
        risk, event_time = [np.asarray(x) for x in [risk, event_time]]

        # sort by event_time
        event_order = event_time.argsort().astype('int')
        sorted_event_time = event_time[event_order]
        sorted_risk = risk[event_order]

        # censoring
        censor_vect = sorted_event_time < censor_thres

        top = bot = 0
        for i in range(len(risk)):
            times_truth = sorted_event_time > sorted_event_time[i]
            risk_truth = sorted_risk < sorted_risk[i]

            i_top = times_truth & risk_truth
            i_top = i_top & censor_vect
            i_bot = times_truth
            i_bot = i_bot & censor_vect

            top += i_top.sum()
            bot += i_bot.sum()

        return top/float(bot)





        pass

    def overload_dataloader(self, loader):
        self._data_loader = loader