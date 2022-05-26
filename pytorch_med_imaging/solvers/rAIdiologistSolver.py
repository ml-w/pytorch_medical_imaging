import os
import torch

import pandas as pd
import numpy as np
from .BinaryClassificationSolver import BinaryClassificationSolver
from ..loss import ConfidenceBCELoss

__all__ = ['rAIdiologistSolver']

class rAIdiologistSolver(BinaryClassificationSolver):
    def __init__(self, *args, **kwargs):
        super(rAIdiologistSolver, self).__init__(*args, **kwargs)

        rAIdiologist_kwargs = {
            'rAI_fixed_mode': None,
            'rAI_conf_weight': 0.2,
            'rAI_conf_weight_scheduler': 'default'
        }
        config = kwargs['config']
        self._total_num_epoch = int(config['RunParams'].get('num_of_epochs'))
        self._rAIdiologist_kwargs = self._load_default_attr(rAIdiologist_kwargs)
        self._current_mode = self.rAI_fixed_mode

        #TODO: port conf_factor parameter here
        self.set_loss_function(ConfidenceBCELoss())

        if os.getenv('CUBLAS_WORKSPACE_CONFIG') not in [":16:8", ":4096:2"]:
            self._logger.warning(f"Env variable CUBLAS_WORKSPACE_CONFIG was not set properly, which may invalidate"
                                 f" deterministic behavior of LSTM.")

    # TODO: Add scheduler to schedule the loss function factor

    def _build_validation_df(self, g, res):
        r"""Tailored for rAIdiologist, model output were of shape (B x 3), where the first element is
        the prediction, the second element is the confidence and the third is irrelevant and only used
        by the network."""

        # res: (B x C), g: (B x 1)
        _data =np.concatenate([res.squeeze().data.cpu().numpy(), g.data.cpu().numpy()], axis=-1)
        _df = pd.DataFrame(data=_data, columns=['res_%i'%i for i in range(res.shape[-1])] + ['g'])

        # model_output: (B x num_class + 1)
        dic = torch.zeros_like(res[..., :-1])
        dic = dic.type_as(res).int() # move to cuda if required
        dic[torch.where(res[..., :-1] >= 0.5)] = 1
        return _df, dic.view(-1, 1)

    def _align_g_res_size(self, g, res):
        # g: (B x 1), res is not important here
        g = g.squeeze()
        return g.view(-1, 1)

    def _epoch_prehook(self, *args, **kwargs):
        r"""Update mode of network"""
        super(rAIdiologistSolver, self)._epoch_prehook(*args, **kwargs)
        current_epoch = self.plotter_dict.get('epoch_num', 0)
        total_epoch = self._total_num_epoch

        if self.rAI_fixed_mode is None:
            # mode is scheduled to occupy 25% of all epochs
            epoch_progress = current_epoch / float(total_epoch)
            current_mode = min(int(epoch_progress * 4) + 1, 4)
        else:
            current_mode = int(self.rAI_fixed_mode)
        if not current_mode == self._current_mode:
            self._logger.info(f"Setting rAIdiologist mode to {current_mode}")
            self._current_mode = current_mode
            if isinstance(self.net, torch.nn.DataParallel):
                self.net.get_submodule('module').set_mode(self._current_mode)
            else:
                self.net.set_mode(self._current_mode)

    def _epoch_callback(self, *args, **kwargs):
        super(rAIdiologistSolver, self)._epoch_callback(*args, **kwargs)
        current_epoch = self.plotter_dict.get('epoch_num', None)
        total_epoch = self._total_num_epoch

        if current_mode in (3, 4):
            # step conf loss function scheduler
            if isinstance(self.lossfunction, ConfidenceBCELoss):
                if self.rAI_conf_weight_scheduler == 'default':
                    if self.lossfunction.conf_factor == 0: # generally in mode 1 & 2, confidence is ignored
                        self.lossfunction.conf_factor = 0.1
                    self.lossfunction.conf_factor = max(self.lossfunction.conf_factor * 1.01, 0.5)



