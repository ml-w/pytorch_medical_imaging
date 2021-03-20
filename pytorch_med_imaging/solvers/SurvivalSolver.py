from .SolverBase import SolverBase
from ..logger import Logger
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gc

from torch import optim
from torch.utils import checkpoint
from ..loss import CoxNLL

import tqdm.auto as auto

__all__ = ['SurvivalSolver']

class SurvivalSolver(SolverBase):
    def __init__(self, _, __, net, param_optim, param_iscuda, grad_iter:int = 5,
                 censor_value:float =-1, param_initWeight=None, logger=None, config=None):
        """
        This class is a special solver for survival analysis. This class uses negative log cox harzard as loss
        function as default. Because of that, the gradient propagation can only happen after a the whole set of
        data passed through the forward flow, which would require a lot of GRAM to store the gradients. Therefore,
        only a designated amount of steps are forward with gradient computation. This can be specified by `grad_iter`.
        """

        assert isinstance(logger, Logger) or logger is None, "Logger incorrect settings!"

        if logger is None:
            logger = Logger[self.__class__.__name__]

        self._decay_init_weight = param_initWeight if not param_initWeight is None else 0
        self._grad_iter = grad_iter
        self._censor_value = censor_value
        self._config = config
        self._logger = Logger[__class__.__name__]

        if not self._config is None:
            self._censor_value = self._get_params_from_solver_config('censor_value', 5, True)
            self._grad_iter = self._get_params_from_solver_config('grad_iter', 2, True)

        solver_configs = {}

        # Check network
        if not hasattr(net, 'forward'):
            # terminate if not correctly specified
            logger.critical(f"Network is not correctly specified. Network: {net}")
            raise ArithmeticError("Network is not correctly specified in solver.")

        # Create optimizer and loss function
        lossfunction = CoxNLL(self._censor_value)
        # optimizer = optim.SGD(net.parameters(), lr=param_optim['lr'], momentum=param_optim['momentum'])
        optimizer = optim.Adam(net.parameters(), lr=param_optim['lr'])
        iscuda = param_iscuda
        if param_iscuda:
            lossfunction = lossfunction.cuda()
            net = net.cuda()

        solver_configs['optimizer'] = optimizer
        solver_configs['lossfunction'] = lossfunction
        solver_configs['net'] = net
        solver_configs['iscuda'] = iscuda
        solver_configs['logger'] = logger

        super(SurvivalSolver, self).__init__(solver_configs)

    def validation(self):
        if self._data_loader_val is None:
            self._logger.warning("Validation skipped because no loader is available.")
            return []

        self._logger.info("Start validation...")
        with torch.no_grad():
            self._net = self._net.eval()
            net_out = []
            G = []

            for s, g in auto.tqdm(self._data_loader_val, desc="Validation", position=2):
                out, g = self._feed_forward(s, g)
                while g.dim() < 2:
                    g = g.unsqueeze()

                # self._logger.debug(f"outshape: {out.shape}, gshape: {g.shape}")
                net_out.append(out)
                G.append(g)

            net_out = torch.cat(net_out, 0)
            try:
                G = torch.cat(G,0)
            except Exception as e:
                self._logger.exception(e)
                self._logger.info(f"G: {G}")

            # compute C-index
            censor_vect = (G[:,:-1].cpu().numpy() < self._censor_value).flatten() \
                          & G[:,-1].cpu().numpy().astype('bool').flatten()
            c_index = self._compute_concordance(net_out.cpu().numpy(),
                                                G[:,:-1].cpu().numpy(),
                                                censor_vect
                                                )

            val_loss = self._loss_eval(net_out, G)
            self._logger.debug(f"Validation Result - VAL: {val_loss:.05f} VAL C-index: {c_index:.05f}")
            self.plotter_dict['scalars']['Loss/Validation Loss'] = val_loss.cpu().data
            self.plotter_dict['scalars']['Perf/Validation C-index'] = c_index

    def train_set_validation(self):
        if self._data_loader_val is None:
            self._logger.warning("Validation skipped because no loader is available.")
            return []

        self._logger.info("Start validation...")
        with torch.no_grad():
            self._net = self._net.eval()
            net_out = []
            G = []

            for s, g in auto.tqdm(self._data_loader, desc="Validation", position=2):
                out, g = self._feed_forward(s, g)
                while g.dim() < 2:
                    g = g.unsqueeze()

                # self._logger.debug(f"outshape: {out.shape}, gshape: {g.shape}")
                net_out.append(out)
                G.append(g)

            net_out = torch.cat(net_out, 0)
            try:
                G = torch.cat(G,0)
            except Exception as e:
                self._logger.exception(e)
                self._logger.info(f"G: {G}")

            # compute C-index
            censor_vect = (G[:,:-1].cpu().numpy() < self._censor_value).flatten() \
                          & G[:,-1].cpu().numpy().astype('bool').flatten()
            c_index = self._compute_concordance(net_out.cpu().numpy(),
                                                G[:,:-1].cpu().numpy(),
                                                censor_vect)

            self._logger.debug(f"Training C-index: {c_index:.05f}")
            self.plotter_dict['scalars']['Perf/Training C-index'] = c_index




    def _feed_forward(self, *args):
        s, g = args
        try:
            s = self._match_type_with_network(s)
            g = self._match_type_with_network(g)
        except:
            self._logger.exception("Failed to match input to network type. Falling back.")
            if self._iscuda:
                s = self._force_cuda(s)
                g = self._force_cuda(g)
                self._logger.debug("_force_cuda() typed data as: {}".format(
                    [ss.dtype for ss in s] if isinstance(s, list) else s.dtype))

        if isinstance(s, list):
            out = self._net.forward(*s)
            # out = checkpoint.checkpoint(self._net.forward, *s)
        else:
            out = self._net.forward(s)
            # out = checkpoint.checkpoint(self._net.forward, s)
        return out, g

    def _loss_eval(self, *args):
        out, G = args
        loss = self._lossfunction(out, G[:, :-1], G[:,-1]) # last column reserved for event status
        return loss

    def __dep_step(self, *args):
        """
        Optimizer step after each epoch instead of each iteration step, so this is reduced to only
        doing network forward.
        """
        out = self._feed_forward(*args)
        self._called_time += 1
        return out

    def step(self, *args):
        out, g = self._feed_forward(*args)
        loss = self._loss_eval(out, g)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._called_time += 1
        return out, loss.cpu().data

    def solve_epoch(self, epoch_number):
        """
        Because this solver is fixed to use Cox harzard loss, the gradients only computed after
        each epoch, and this requires more memory to do.
        """
        if self._data_loader_val is None:
            self._logger.error("Dataloader not found!")
            raise AttributeError("You have not properly setup the dataloader")

        E = []
        # Reset dict each epoch
        self._net.train()
        self.plotter_dict = {'scalars': {}, 'epoch_num': epoch_number}
        grad_itered = 0

        for step_idx, samples in enumerate(self._data_loader):
            s, g = samples

            # Check if all elements in the batch are censored. Last column of gt reserved for event_status
            if all(list(g[:,-1] == False)):
                self._logger.info(f"\t[Step {step_idx:04d}] Skipped")
                continue

            out, loss = self.step(s, g)

            E.append(loss.data.cpu())
            self._logger.info("\t[Step %04d] Loss: %.05f"%(step_idx, loss))

            grad_itered += 1
            # Display epoch results
            _pairs = zip(out.flatten().data.cpu(), g[:,:-1].flatten().data.cpu())
            _df = pd.DataFrame(_pairs, columns=['res', 'g'], dtype=float)
            self._logger.debug('\n' + _df.to_string())
            del _pairs, _df

            del s, g, out, loss
            gc.collect()

        epoch_loss = np.array(E).mean()
        self.plotter_dict['scalars']['Loss/Loss'] = epoch_loss

        self._logger.info("Initiating validation.")
        self.validation()
        self.train_set_validation()
        self._epoch_callback()
        self.decay_optimizer(epoch_loss)

    @staticmethod
    def _compute_concordance(risk, event_time, censor_vect):
        r"""
        Compute the concordance index. Assume no ties.

        .. math::

            $C-index = \frac{\sum_{i,j} I[T_j < T_i] \cdot I [\eta_j > \eta_i] d_j}{1}$

        """
        # convert everything to numpy
        risk, event_time = [np.asarray(x) for x in [risk, event_time]]

        # Logger[__class__.__name__].error(f"risk: {risk.shape}")
        # Logger[__class__.__name__].error(f"censor: {censor_vect.shape}")
        # Logger[__class__.__name__].error(f"eventime: {event_time.shape}")

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
            Logger[__class__.__name__].warning("Got nan when computing concordance. Replace by 0.")
            c_index = 0

        return np.clip(c_index, 0, 1)


