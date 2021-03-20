import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tqdm import *
from abc import abstractmethod

import numpy as np
import gc
from logging import Logger

class SolverBase(object):
    """

    Args:
        solver_configs (dict):
            Child class should prepare the configuration. Some keys are compulsary.

    Kwargs:
        'net_init': Initialization method. (Not implemented)

    Attributes:
        _optimizer (torch.optim.Optimizer):
            This is the optimizer.
        plotter_dict (dict):
            This dict could be used by the child class to perform plotting after validation or in each step.

    """
    def __init__(self, solver_configs, **kwargs):
        super(SolverBase, self).__init__()

        # required
        self._optimizer         = solver_configs['optimizer']
        self._lossfunction      = solver_configs['lossfunction']
        self._net               = solver_configs['net']
        self._iscuda            = solver_configs['iscuda']

        # optional
        self._logger            = solver_configs['logger'] if 'logger' in solver_configs else \
                                                                        Logger[self.__class__.__name__]

        # Optimizer
        self._lr_decay          = solver_configs['lrdecay'] if 'lrdecay' in solver_configs else None
        self._mom_decay         = solver_configs['momdecay'] if 'momdecay' in solver_configs else None
        self._lr_decay_func     = lambda epoch: np.exp(-self._lr_decay * epoch)
        self._mom_decay_func    = lambda mom: np.max(0.2, mom * np.exp(-self._mom_decay))
        self._lr_schedular      = None
        self._called_time = 0
        self._decayed_time= 0

        # Added config not used in base class
        self._config            = kwargs.get('config', None)

        # internal_attributes
        self._net_weight_type   = None
        self._data_logger       = None
        self._data_loader       = None
        self._data_loader_val   = None
        self._tb_plotter        = None

        # external_att
        self.plotter_dict      = {}


        self._logger.info("Solver were configured with options: {}".format(solver_configs))
        if  len(kwargs):
            self._logger.warning("Some solver configs were not used: {}".format(kwargs))


    def get_net(self):
        if torch.cuda.device_count() > 1:
            try:
                return self._net.module
            except AttributeError:
                return self._net
        else:
            return self._net

    def get_optimizer(self):
        return self._optimizer

    def set_lr_decay(self, decay):
        self._lr_decay = decay

    def set_lr_decay_exp(self, decay):
        self._lr_decay = decay
        self._lr_schedular = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, self._lr_decay)

    def set_lr_decay_func(self, func):
        assert callable(func), "Insert function not callable!"
        self._lr_decay_func = func
        self._lr_schedular = torch.optim.lr_scheduler.LambdaLR(self._optimizer, self._lr_decay_func)

    def set_dataloader(self, dataloader, data_loader_val=None):
        self._data_loader = dataloader
        self._data_loader_val = data_loader_val

    def set_lr_decay_to_reduceOnPlateau(self, default_patience, factor, **kwargs):
        _default_kwargs = {
            'factor': factor,
            'patience': int(default_patience),
            'cooldown':2,
            'min_lr': 1E-6,
            'threshold':0.05,
            'threshold_mode':'rel'
        }
        try:
            for keys in kwargs:
                _default_kwargs[keys] = kwargs[keys]
        except:
            self._logger.warning("Extraction of parameters failed. Retreating to use default.")

        self._logger.debug("Set lr_scheduler to decay on plateau with params: {}.".format(_default_kwargs))
        self._lr_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            **_default_kwargs
        )

    def set_momentum_decay(self, decay):
        self._mom_decay = decay

    def set_momentum_decay_func(self, func):
        assert callable(func), "Insert function not callable!"
        self._mom_dcay_func = func

    def set_plotter(self, plotter):
        self._tb_plotter = plotter

    def net_to_parallel(self):
        if torch.cuda.device_count()  > 1:
            self._net = nn.DataParallel(self._net)

    def set_loss_function(self, func: callable):
        self._logger.debug("loss functioning override.")
        self._lossfunction = func


    def step(self, *args):
        out = self._feed_forward(*args)
        loss = self._loss_eval(out, *args)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._called_time += 1
        return out, loss.cpu().data

    def decay_optimizer(self, *args):
        if not self._lr_schedular is None:
            try:
                self._lr_schedular.step(*args)
            except:
                self._lr_schedular.step()
        if not self._mom_decay is None:
            for pg in self._optimizer.param_groups:
                pg['momentum'] = self._mom_decay_func(pg['momemtum'])
        self._decayed_time += 1
        self._log_print("Decayed optimizer...")

    def inference(self, *args):
        with torch.no_grad():
            out = self._net.forward(*list(args))
        return out

    def solve_epoch(self, epoch_number):
        """
        Run this per epoch.
        """
        E = []
        # Reset dict each epoch
        self._net.train()
        self.plotter_dict = {'scalars': {}, 'epoch_num': epoch_number}
        for step_idx, samples in enumerate(self._data_loader):
            s, g = samples

            # initiate one train step. Things should be plotted in decorator of step if needed.
            out, loss = self.step(s, g)

            E.append(loss.data.cpu())
            self._logger.info("\t[Step %04d] loss: %.010f"%(step_idx, loss.data))

            self._step_callback(s, g, out, loss, step_idx=step_idx)
            del s, g, out, loss
            gc.collect()

        epoch_loss = np.array(E).mean()
        self.plotter_dict['scalars']['Loss/Loss'] = epoch_loss

        self._logger.info("Initiating validation.")
        self.validation()
        self._epoch_callback()
        self.decay_optimizer(epoch_loss)


    @abstractmethod
    def validation(self, *args, **kwargs):
        """
        This is called after each epoch.
        """
        raise NotImplementedError("Validation is not implemented in this solver.")


    def _log_print(self, msg, level=20):
        if not self._logger is None:
            try:
                self._logger.log(level, msg)
                tqdm.write(msg)
            except:
                tqdm.write(msg)


    def _match_type_with_network(self, tensor):
        """
        Return a tensor with the same type as the first weight of `self._net`. This function seems to cause CUDA
        error in pytorch 1.3.0

        Args:
            tensor (torch.Tensor or list): Input `torch.Tensor` or list of `torch.Tensor`

        Returns:
            out (torch.Tensor)
        """
        assert isinstance(tensor, list) or torch.is_tensor(tensor) or isinstance(tensor, tuple), \
            "_match_type_with_network: input type error! Got type: {}".format(tensor)

        for name, module in self._net.named_modules():
            try:
                self._net_weight_type = module.weight.type()
                #self._logger.debug("Module type is: {}".format(self._net_weight_type))
                break
            except AttributeError:
                continue
            except Exception as e:
                self._logger.log_print_tqdm("Unexpected error in type convertion of solver")

        if self._net_weight_type is None:
            # In-case type not found
            self._logger.log_print_tqdm("Cannot identify network type, falling back to float type.")
            self._net_weight_type = torch.FloatTensor

        # Do nothing if type is already correct.
        try:
            if isinstance(tensor, list) or isinstance(tensor, tuple):
                if all([t.type() == self._net_weight_type for t in tensor]):
                    return tensor
            else:
                if tensor.type() == self._net_weight_type:
                    return tensor
        except:
            self._logger.warning("Can't determine if type is already followed.")

        # We also expect list input too.
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            out = []
            for ss in tensor:
                try:
                    out.append(ss.type(self._net_weight_type))
                except:
                    out.append(ss)
        else:
            out = tensor.type(self._net_weight_type)
        return out

    @staticmethod
    def _force_cuda(arg):
        return [a.float().cuda() for a in arg] if isinstance(arg, list) else arg.cuda()

    @abstractmethod
    def _feed_forward(self, *args):
        raise NotImplementedError

    @abstractmethod
    def _loss_eval(self, *args):
        raise NotImplementedError

    @abstractmethod
    def _step_callback(self, s, g, out, loss, step_idx=None):
        return

    def _epoch_callback(self, *args, **kwargs):
        """
        Default callback
        """
        scalars = self.plotter_dict.get('scalars', None)
        writer_index = self.plotter_dict.get('epoch_num', None)

        if scalars is None:
            return
        elif self._tb_plotter is None:
            return
        else:
            try:
                self._tb_plotter.plot_scalars(writer_index, scalars)
                self._tb_plotter.plot_weight_histogram(self._net, writer_index)
            except:
                self._logger.exception("Error occured in default epoch callback.")

    def _get_params_from_config(self, section, key, default=None, with_eval=False):
        try:
            if with_eval:
                out = self._config[section].get(key, default)
                return eval(out) if isinstance(out,str) else out
            else:
                return self._config[section].get(key, default)
        except AttributeError:
            self._logger.warning(f"Key absent in config: ({section},{key})")
            return default
        except:
            self._logger.exception(f"Unexpected error when reading params with key: ({section}, {key})")
            return default

    def _get_params_from_solver_config(self, key, default=None, with_eval=False):
        return self._get_params_from_config('SolverParams', key, default, with_eval)


class SolverEarlyStopScheduler(object):
    def __init__(self, configs):
        super(SolverEarlyStopScheduler, self).__init__()
        self._configs = configs
        self._logger = Logger[__class__.__name__]
        self._last_loss = 1E-32
        self._watch = 0

        if self._configs is None:
            self._warmup = None
            self._patience = None
            pass
        else:
            _c = self._configs['RunParams'].get('early_stop', {})
            _c = eval(_c)

            if not isinstance(_c, dict):
                self._logger.error(f"Wrong early stopping settings, cannot eval into dict. Receive arguments: {_c}")
                self._logger.warning("Ignoring early stopping options")
                self._configs = None
                return

            warmup = _c.get('warmup', None)
            patience = _c.get('patience', None)

            if warmup is None or patience is None or warmup < 0 or patience < 0:
                self._logger.warning(f"Wrong ealry stopping settings: {_c}")
                self._logger.warning("Ignoring early stopping options")
                self._configs = None
                return

            self._warmup = warmup
            self._patience = patience


    def step(self,
             loss: float,
             epoch: int):
        r"""
        Returns 1 if reaching stopping criteria, else 0.
        """
        # ignore if there are no configs
        if self._configs is None:
            return 0
        else:
            if epoch < self._warmup:
                return 0
            else:
                # reset if lass is smaller than last loss
                if loss < self._last_loss:
                    self._watch = 0
                    self._last_loss = loss
                    return 0
                else:
                    self._watch += 1

            # Stop if enough iterations show no decrease
            if self._watch > self._patience:
                return 1
