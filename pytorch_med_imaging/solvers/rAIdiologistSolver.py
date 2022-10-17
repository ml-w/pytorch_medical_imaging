import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .BinaryClassificationSolver import BinaryClassificationSolver
from ..loss import ConfidenceBCELoss
import gc

__all__ = ['rAIdiologistSolver']

class rAIdiologistSolver(BinaryClassificationSolver):
    def __init__(self, *args, **kwargs):
        super(rAIdiologistSolver, self).__init__(*args, **kwargs)

        self._current_mode = None # initially, the mode is unsetted
        if Path(self.solverparams_rai_pretrained_swran).is_file():
            self._logger.info(f"Loading pretrained SWRAN network from: {self.solverparams_rai_pretrained_swran}")
            self.net.load_pretrained_swran(self.solverparams_rai_pretrained_swran)
        else:
            self._logger.warning(f"Pretrained SWRAN network specified ({self.solverparams_rai_pretrained_swran}) "
                                 f"but not loaded.")

        if os.getenv('CUBLAS_WORKSPACE_CONFIG') not in [":16:8", ":4096:2"]:
            self._logger.warning(f"Env variable CUBLAS_WORKSPACE_CONFIG was not set properly, which may invalidate"
                                 f" deterministic behavior of LSTM.")

        # Turn off record
        self.get_net().RECORD_ON = False

    def _load_default_attr(self, default_attr):
        _default_attr = {
            'solverparams_rai_fixed_mode': None,
            'solverparams_rai_pretrained_swran': "",
            'solverparams_rai_classification': False
        }
        if isinstance(default_attr, dict):
            _default_attr.update(default_attr)
        super(rAIdiologistSolver, self)._load_default_attr(_default_attr)

    def create_lossfunction(self):
        if not self.solverparams_rai_classification:
            super(rAIdiologistSolver, self).create_lossfunction()
        else:
            super(BinaryClassificationSolver, self).create_lossfunction()

    def _build_validation_df(self, g, res):
        r"""Tailored for rAIdiologist, model output were of shape (B x 3), where the first element is
        the prediction, the second element is the confidence and the third is irrelevant and only used
        by the network. In mode 0, the output shape is (B x 1)"""

        # res: (B x C)/(B x 1), g: (B x 1)
        chan = res.shape[-1] # if chan > 1, there is a value for confidence
        _data =np.concatenate([res.view(-1, chan).data.cpu().numpy(), g.data.view(-1, 1).cpu().numpy()], axis=-1)
        _df = pd.DataFrame(data=_data, columns=['res_%i'%i for i in range(chan)] + ['g'])
        _df['Verify_wo_conf'] = (_df['res_0'] >= 0.5) == (_df['g'] > 0)
        _df['Verify_wo_conf'].replace({True: "Correct", False: "Wrong"}, inplace=True)
        if chan > 1:
            _df['Verify_w_conf'] = ((_df['res_0'] >= 0.5) == (_df['g'] > 0)) == (_df['res_1'] >= 0.5)
            _df['Verify_w_conf'].replace({True: "Correct", False: "Wrong"}, inplace=True)

        # res: (B x C)/(B x 1)
        if chan > 1:
            dic = torch.zeros_like(res[..., :-1])
            dic = dic.type_as(res).int() # move to cuda if required
            dic[torch.where(res[..., :-1] >= 0.5)] = 1
        else:
            dic = (res >= 0.5).type_as(res).int()
        return _df, dic.view(-1, 1)

    def _align_g_res_size(self, g, res):
        # g: (B x 1), res is not important here
        g = g.squeeze()
        return g.view(-1, 1)

    def _epoch_prehook(self, *args, **kwargs):
        r"""Update mode of network"""
        super(rAIdiologistSolver, self)._epoch_prehook(*args, **kwargs)
        current_epoch = self.plotter_dict.get('epoch_num', 0)
        total_epoch = self.solverparams_num_of_epochs

        # Schedule mode of the network and findout if new mode is needed
        if self.solverparams_rai_fixed_mode is None:
            # mode is scheduled to occupy 25% of all epochs
            epoch_progress = current_epoch / float(total_epoch)
            current_mode = min(int(epoch_progress * 4) + 1, 4)
        else:
            current_mode = int(self.solverparams_rai_fixed_mode)

        # If new mode is needed, change mode
        if not current_mode == self._current_mode:
            self._logger.info(f"Setting rAIdiologist mode to {current_mode}")
            self._current_mode = current_mode
            if isinstance(self.net, torch.nn.DataParallel):
                self.net.get_submodule('module').set_mode(self._current_mode)
            else:
                self.net.set_mode(self._current_mode)

    def _epoch_callback(self, *args, **kwargs):
        r"""
        Used to fine tune loss weights here
        """
        super(rAIdiologistSolver, self)._epoch_callback(*args, **kwargs)
        current_epoch = self.plotter_dict.get('epoch_num', None)
        total_epoch = self.solverparams_num_of_epochs

    def solve_epoch(self, epoch_number):
        """
        Modify this to expect ArithmeticError from step
        """
        self._epoch_prehook()
        E = []
        # Reset dict each epoch
        self.net.train()
        self.plotter_dict = {'scalars': {}, 'epoch_num': epoch_number}
        for step_idx, mb in enumerate(self._data_loader):
            s, g = self._unpack_minibatch(mb, self.solverparams_unpack_keys_forward)

            # initiate one train step. Things should be plotted in decorator of step if needed.
            try:
                out, loss = self.step(s, g)
            except ArithmeticError:
                self._logger.warning(f"Recovering from error when dealing with: {mb['uid']}, "
                                     f"skipping step {step_idx:04d}")
                self.net.zero_grad()
                del s, g
                continue

            E.append(loss.data.cpu())
            self._logger.info("\t[Step %04d] loss: %.010f"%(step_idx, loss.data))

            self._step_callback(s, g, out.cpu().float(), loss.data.cpu(),
                                step_idx=epoch_number * len(self._data_loader) + step_idx)
            del s, g, out, loss, mb
            gc.collect()

        epoch_loss = np.array(E).mean()
        self.plotter_dict['scalars']['Loss/Loss'] = epoch_loss
        self._last_epoch_loss = epoch_loss

        self._logger.info("Initiating validation.")
        self._last_val_loss = self.validation()
        self._epoch_callback()
        self.decay_optimizer(epoch_loss)

    def validation(self):
        if not self.solverparams_rai_classification:
            return super(rAIdiologistSolver, self).validation()
        else:
            return super(BinaryClassificationSolver, self).validation()

    def _loss_eval(self, *args):
        if not self.solverparams_rai_classification:
            return super(rAIdiologistSolver, self)._loss_eval(*args)
        else:
            return super(BinaryClassificationSolver, self)._loss_eval(*args)