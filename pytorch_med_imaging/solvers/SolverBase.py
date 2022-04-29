import torch
import torch.nn as nn
import torchio as tio

from ..loss import *

from tqdm import *
from abc import abstractmethod

import numpy as np
import gc
import ast
from mnts.mnts_logger import MNTSLogger

class SolverBase(object):
    """
    Args:
        solver_configs (dict):
            Child class should prepare the configuration. Some keys are compulsory.
        **kwargs

    Attributes:
        optimizer (nn.Module):
            Optimizer for training.
        lossfunction (nn.Module):
            Loss function.
        net (str or nn.Module):

        plotter_dict (dict):
            This dict could be used by the child class to perform plotting after validation or in each step.

    """
    def __init__(self, solver_configs: dict, **kwargs):
        super(SolverBase, self).__init__()

        # these attributes requires evaluation and cannot be get directly from the ini file.
        default_attr = {
            'optimizer':        None,   # required
            'lossfunction':     None,   # required
            'net':              None,   # required
            'iscuda':           None,   # required
            'lr_decay':         None,
            'mom_decay':        None,
            'lr_decay_func':    lambda epoch: np.exp(-self.lr_decay * epoch),
            'mom_decay_func':   lambda mom: np.max(0.2, mom * np.exp(-self.mom_decay)),
            'lr_schedular':     None
        }
        default_attr.update((k, solver_configs[k]) for k in default_attr.keys() & solver_configs.keys())
        required_att = ('optimizer', 'lossfunction', 'net', 'iscuda')

        if any([default_attr[k] is None for k in required_att]):
            self._logger.error("Some required attributes are not specified.")
            raise AttributeError(f"Must specify these attributes: {','.join(required_att)}")
        self.__dict__.update(default_attr)

        # optional
        self._logger            = solver_configs.get('logger', None)
        if self._logger is None:
            self._logger        = MNTSLogger[self.__class__.__name__]

        # Optimizer attributies
        self._called_time       = 0
        self._decayed_time      = 0

        # Added config not used in base class
        self._config            = kwargs.get('config', None)

        # internal_attributes
        self._net_weight_type   = None
        self._data_logger       = None
        self._data_loader       = None
        self._data_loader_val   = None
        self._tb_plotter        = None
        self._local_rank        = 0

        # external_att
        self.plotter_dict      = {}

        # create loss function if not specified
        if self.lossfunction is None:
            self._logger.info("Trying to create loss function.")
            self.create_lossfunction()

        self._logger.info("Solver were configured with options: {}".format(solver_configs))
        if  len(kwargs):
            self._logger.warning("Some solver configs were not used: {}".format(kwargs))

    def _load_default_attr(self, default_dict):
        r"""
        Load default dictionary as attr from ini config, [SolverParams] section.
        """
        final_dict = {}
        for key in default_dict:
            val = default_dict[key]
            if isinstance(val, bool):
                final_dict[key] = self._get_params_from_solver_config_with_boolean(key, default_dict[key])
            elif isinstance(val, str):
                final_dict[key] = self._get_params_from_solver_config(key, default_dict[key])
            else:
                final_dict[key] = self._get_params_from_solver_config(key, default_dict[key], with_eval=True)

        self._logger.debug(f"final_dict: {final_dict}")
        self.__dict__.update(final_dict)

    def get_net(self):
        if torch.cuda.device_count() > 1:
            try:
                return self.net.module
            except AttributeError:
                return self.net
        else:
            return self.net

    def get_optimizer(self):
        return self.optimizer

    def set_lr_decay(self, decay):
        self.lr_decay = decay

    def set_lr_decay_exp(self, decay):
        self.lr_decay = decay
        self.lr_schedular = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.lr_decay)

    def set_lr_decay_func(self, func):
        assert callable(func), "Insert function not callable!"
        self.lr_decay_func = func
        self.lr_schedular = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_decay_func)

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
            _default_kwargs.update(kwargs)
        except:
            self._logger.warning("Extraction of parameters failed. Retreating to use default.")

        self._logger.debug("Set lr_scheduler to decay on plateau with params: {}.".format(_default_kwargs))
        self.lr_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            **_default_kwargs
        )

    def set_momentum_decay(self, decay):
        self.mom_decay = decay

    def set_momentum_decay_func(self, func):
        assert callable(func), "Insert function not callable!"
        self._mom_dcay_func = func

    def set_plotter(self, plotter):
        self._tb_plotter = plotter

    def net_to_parallel(self):
        if (torch.cuda.device_count()  > 1) & self.iscuda:
            self._logger.info("Multi-GPU detected, using nn.DataParallel for distributing workload.")
            self.net = nn.DataParallel(self.net)

    def set_loss_function(self, func: callable):
        self._logger.debug("loss functioning override.")
        # Check if its cuda mode
        is_cuda = self.lossfunction.is_cuda
        del self.lossfunction
        if is_cuda:
            self.lossfunction = func.cuda()
        else:
            self.lossfunction = func

    def create_lossfunction(self, *args, **kwargs):
        r"""
        Try to create loss function from parameters specified by the config file.
        """
        self._logger.info("Creating loss function from config specification.")

        # Follows the specification in config
        _lossfunction = eval(self._get_params_from_solver_config('loss_func', None))

        # Extract parameters
        try:
            _loss_params = dict(self._config['LossParams'])
        except KeyError:
            _loss_params = {}
        except Exception as e:
            self._logger.exception("Unknown error when creating loss function.")
            raise RuntimeError("Can't proceed.")

        # Try to eval all of the arguments
        for keys in _loss_params:
            try:
                _loss_params[keys] = ast.literal_eval(_loss_params[keys])
            except:
                self._logger.exception(f"Failed to eval key: {keys}")


        # Create loss function accordingly
        if not _lossfunction is None and issubclass(_lossfunction, nn.Module):
            self.lossfunction = _lossfunction(**_loss_params)
            return self.lossfunction
        else:
            self._logger.warning(f"Cannot create loss function using: {_lossfunction}")
            return None

    def step(self, *args):
        out = self._feed_forward(*args)
        loss = self._loss_eval(out, *args)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._called_time += 1
        return out, loss.cpu().data

    def create_optimizer(self,
                         net_params,
                         param_optim) -> torch.optim:
        r"""
        Create optimizer based on provided specifications
        Args:
            net_params:
            param_optim:

        Returns:

        """
        if self.optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(net_params, lr=param_optim['lr'])
        elif self.optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(net_params, lr=param_optim['lr'],
                                  momentum=param_optim['momentum'])
        else:
            raise AttributeError(f"Expecting optimzer to be one of ['Adam'|'SGD']")
        return optimizer

    def decay_optimizer(self, *args):
        if not self.lr_schedular is None:
            try:
                self.lr_schedular.step(*args)
            except:
                self.lr_schedular.step()
        if not self.mom_decay is None:
            for pg in self.optimizer.param_groups:
                pg['momentum'] = self.mom_decay_func(pg['momemtum'])
        self._decayed_time += 1
        self._log_print("Decayed optimizer...")

    def inference(self, *args):
        with torch.no_grad():
            out = self.net.forward(*list(args))
        return out

    def solve_epoch(self, epoch_number):
        """
        Run this per epoch.
        """
        E = []
        # Reset dict each epoch
        self.net.train()
        self.plotter_dict = {'scalars': {}, 'epoch_num': epoch_number}
        for step_idx, mb in enumerate(self._data_loader):
            s, g = self._unpack_minibatch(mb, self.unpack_keys_forward)

            # initiate one train step. Things should be plotted in decorator of step if needed.
            out, loss = self.step(s, g)

            E.append(loss.data.cpu())
            self._logger.info("\t[Step %04d] loss: %.010f"%(step_idx, loss.data))

            self._step_callback(s, g, out.cpu().float(), loss.data.cpu(),
                                step_idx=epoch_number * len(self._data_loader) + step_idx)
            del s, g, out, loss, mb
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

        for name, module in self.net.named_modules():
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
            self._logger.warning(f"Can't determine if type is already followed. Input type is {type(tensor)}")
            self._logger.exception(f"Get error {e}")

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
                self._tb_plotter.plot_weight_histogram(self.net, writer_index)
            except:
                self._logger.exception("Error occured in default epoch callback.")

    def _get_params_from_config(self, section, key, default=None, with_eval=False, with_boolean=False):
        try:
            if with_eval:
                if with_boolean:
                    out = self._config[section].getboolean(key, default)
                    return out
                else:
                    out = self._config[section].get(key, default)
                    return ast.literal_eval(out) if isinstance(out,str) else out
            else:
                return self._config[section].get(key, default)
        except AttributeError:
            self._logger.warning(f"Key absent in config: ({section},{key})")
            self._logger.exception('Exception: ')
            return default
        except:
            self._logger.exception(f"Unexpected error when reading params with key: ({section}, {key})")
            return default

    def _get_params_from_solver_config(self, key, default=None, with_eval=False):
        return self._get_params_from_config('SolverParams', key, default, with_eval)

    def _get_params_from_solver_config_with_boolean(self, key, default=None):
        return self._get_params_from_config('SolverParams', key, default, True, True)

    def _unpack_minibatch(self, minibatch, unpacking_keys):
        r"""Unpack mini-batch drawn by torchio.Queue or torchio.SubjectsDataset.
        TODO: allow custom modification after unpacking, e.g. concatentation
        !!! If you chnage this you need to also change InferenceBase._unpacking_keys, I know its not ideal but I dont
        plan to open aonther class for just this function
        """
        out = []
        for key in unpacking_keys:
            if isinstance(key, (tuple, list)):
                _out = []
                for kk in key:
                    try:
                        _out.append(minibatch[kk][tio.DATA])
                    except (AttributeError, IndexError):
                        _out.append(minibatch[kk])
                    except Exception as e:
                        self._logger.exception(f"Receive unknown exception during minibactch unpacking for: {key}")
                out.append(tuple(_out))
            else:
                try:
                    out.append(minibatch[key][tio.DATA])
                except (AttributeError, IndexError):
                    out.append(minibatch[key])
                except Exception as e:
                    self._logger.exception(f"Receive unknown exception during minibactch unpacking for: {key}")
        return out


class SolverEarlyStopScheduler(object):
    def __init__(self, configs):
        super(SolverEarlyStopScheduler, self).__init__()
        self._configs = configs
        self._logger = MNTSLogger[__class__.__name__]
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
