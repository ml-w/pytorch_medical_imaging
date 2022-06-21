import ast
import inspect
import re
from abc import abstractmethod

import gc
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import *
from typing import Union, Iterable, Any
from pathlib import Path

import torchio as tio
from ..loss import *

available_lr_scheduler = list(name for name, obj in inspect.getmembers(lr_scheduler) if inspect.isclass(obj))

class SolverBase(object):
    """Base class for all solvers. This class must be inherited before it can work properly. The child
    classes should inherit the abstract methods.

    Args:
        net (torch.nn.Module):
            The network.
        hyperparam_dict (dict):
            This is created by the controller from all options under the section `SolverParams`.
        use_cuda (bool):
            Whether this solver will move the items to cuda for computation.

    Attributes:
        optimizer (nn.Module):
            Optimizer for training.
        lossfunction (nn.Module):
            Loss function.
        plotter_dict (dict):
            This dict could be used by the child class to perform plotting after validation or in each step.

    """
    def __init__(self, net: torch.nn.Module, hyperparam_dict: dict, use_cuda: bool, **kwargs):
        super(SolverBase, self).__init__()

        self.net = net
        self.iscuda = use_cuda
        self.hyperparam_dict = hyperparam_dict
        self.__dict__.update(hyperparam_dict)

        # optional
        self._logger        = MNTSLogger[self.__class__.__name__]

        # Optimizer attributies
        self._called_time       = 0
        self._decayed_time      = 0

        # Added config not used in base class
        # if not hasattr(self, '_config'): # prevent unwanted override
        #     self._config = kwargs.get('config', None)

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
        self._load_default_attr(None)   # inherit in child to default other attributes
        self.create_lossfunction()
        self.create_optimizer(self.net.parameters())

        self._logger.info("Solver were configured with options: {}".format(self.hyperparam_dict))
        if  len(kwargs):
            self._logger.warning("Some solver configs were not used: {}".format(kwargs))

        if self.iscuda:
            self._logger.info("Moving lossfunction and network to GPU.")
            self.lossfunction = self.lossfunction.cuda()
            self.net = self.net.cuda()

    def _load_default_attr(self, default_dict = None):
        r"""If the default_dict items are not specified in the hyperparameter_dict, this will
        load the hyperparameters into __dict__ and self.hyperparameter_dict
        """
        if default_dict is None:
            return

        update_dict = {}
        for key in default_dict:
            if not key in self.__dict__:
                self._logger.debug(f"Loading default value for: {key}")
                update_dict[key] = default_dict[key]
        self.__dict__.update(update_dict)
        self.hyperparam_dict.update(update_dict)

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
        self.solverparams_decay_rate_lr = decay

    def set_lr_decay_exp(self, decay):
        self.solverparams_decay_rate_lr = decay
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.solverparams_decay_rate_lr)

    def set_lr_decay_func(self, func):
        assert callable(func), "Insert function not callable!"
        self.lr_decay_func = func
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_decay_func)

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
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            **_default_kwargs
        )

    def set_lr_scheduler(self, name, *args, **kwargs):
        msg = f"Incorrect lr_scheduler ({name}) specified! Available schedulers are: [{'|'.join(available_lr_scheduler)}]"
        assert name in available_lr_scheduler, msg

        if re.search("^[\W]+", name) is not None:
            raise ArithmeticError(f"Your lr_scheduler setting ({name}) contains illegal characters!")
        sche_class = eval('lr_scheduler.' + name)
        self.lr_scheduler = sche_class(self.optimizer, *args, **kwargs)

    def set_plotter(self, plotter):
        self._tb_plotter = plotter

    def net_to_parallel(self):
        if (torch.cuda.device_count()  > 1) & self.iscuda:
            self._logger.info("Multi-GPU detected, using nn.DataParallel for distributing workload.")
            self.net = nn.DataParallel(self.net)

    def set_loss_function(self, func: torch.nn.Module):
        self._logger.debug("loss functioning override.")
        if self.iscuda:
            try:
                func = func.cuda()
            except:
                self._logger.warning("Failed to move loss function to GPU")
                pass
        del self.lossfunction
        self.lossfunction = func

    def load_checkpoint(self, checkpoint_dir: str):
        if os.path.isfile(self.checkpoint_load):
            # assert os.path.isfile(checkpoint_load)
            try:
                self._logger.info("Loading checkpoint " + checkpoint_load)
                self.get_net().load_state_dict(torch.load(checkpoint_load), strict=False)
            except Exception as e:
                if not self.debug:
                    self._logger.error(f"Cannot load checkpoint from: {checkpoint_load}")
                    raise e
                else:
                    self._logger.warning(f"Cannot load checkpoitn from {checkpoint_load}")
        else:
            self.logger.warning("Checkpoint specified but doesn't exist!")

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
                         net_params
                         ) -> torch.optim:
        r"""
        Create optimizer based on provided specifications
        Args:
            net_params:
            param_optim:

        Returns:

        """
        if self.solverparams_optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(net_params, lr=self.solverparams_decay_rate_lr)
        elif self.solverparams_optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(net_params, lr=self.solverparams_decay_rate_lr,
                                             momentum=self.solverparams_decay_rate_mom)
        else:
            raise AttributeError(f"Expecting optimzer to be one of ['Adam'|'SGD']")
        return self.optimizer

    def decay_optimizer(self, *args):
        if not self.lr_scheduler is None:
            if isinstance(self.lr_scheduler, (lr_scheduler.ReduceLROnPlateau)):
                self.lr_scheduler.step(*args)
            else:
                self.lr_scheduler.step()
        self._decayed_time += 1

        # ReduceLROnPlateau has no get_last_lr attribute
        lass_lr = self.get_last_lr()
        self._logger.debug(f"Decayed optimizer, new LR: {lass_lr}")

    def get_last_lr(self):
        try:
            lass_lr = self.lr_scheduler.get_last_lr()[0]
        except AttributeError:
            if isinstance(self.get_optimizer().param_groups, (tuple, list)):
                lass_lr = self.get_optimizer().param_groups[0]['lr']
            else:
                lass_lr = next(self.get_optimizer().param_groups)['lr']
        except:
            self._logger.warning("Cannot get learning rate!")
            lass_lr = "Error"
        return lass_lr

    def get_last_epoch_loss(self):
        try:
            return self._last_epoch_loss
        except AttributeError:
            return None

    def get_last_val_loss(self):
        try:
            return self._last_val_loss
        except AttributeError:
            return None

    def inference(self, *args):
        with torch.no_grad():
            out = self.net.forward(*list(args))
        return out

    def solve_epoch(self, epoch_number):
        """
        Run this per epoch.
        """
        self._epoch_prehook()
        E = []
        # Reset dict each epoch
        self.net.train()
        self.plotter_dict = {'scalars': {}, 'epoch_num': epoch_number}
        for step_idx, mb in enumerate(self._data_loader):
            s, g = self._unpack_minibatch(mb, self.solverparams_unpack_keys_forward)

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
        self._last_epoch_loss = epoch_loss

        self._logger.info("Initiating validation.")
        self.validation()
        self._epoch_callback()
        self.decay_optimizer(epoch_loss)

    def fit(self, checkpoint_path, debug_validation = False):
        # Error check


        # configure checkpoints
        self.net_to_parallel()
        lastloss = 1e32
        self._logger.info("Start training...")
        for i in range(self.solverparams_num_of_epochs):
            # Skip if --debug-validation flag is true
            if not debug_validation:
                self.solve_epoch(i)
            else:
                self._logger.info("Skip solve_epoch() and directly doing validation.")
                self.plotter_dict['scalars'] = {
                    'Loss/Loss'           : None,
                    'Loss/Validation Loss': None
                }
                self.validation()

            # Prepare values for epoch callback plots
            epoch_loss = self.get_last_epoch_loss()
            val_loss   = self.get_last_val_loss()

            # use validation loss as epoch loss if it exist
            measure_loss = val_loss if val_loss is not None else epoch_loss
            if measure_loss <= lastloss:
                self._logger.info("New loss ({:.03f}) is smaller than previous loss ({:.03f})".format(measure_loss, lastloss))
                self._logger.info("Saving new checkpoint to: {}".format(checkpoint_path))
                self._logger.info("Iteration number is: {}".format(i))
                if not Path(checkpoint_path).parent.is_dir():
                    Path(checkpoint_path).parent.mkdir(parents=True)
                torch.save(self.get_net().state_dict(), checkpoint_path)
                lastloss = measure_loss
                self._logger.info("Update benchmark loss.")
            else:
                torch.save(self.get_net().state_dict(), checkpoint_path.replace('.pt', '_temp.pt'))

            # Save network every 5 epochs
            if i % 5 == 0:
                torch.save(self.get_net().state_dict(), checkpoint_path.replace('.pt', '_{:03d}.pt'.format(i)))

            # TODO: early stopping if criteria true
            # early_stop = earlystopper.step(loss)

            try:
                current_lr = next(self.get_optimizer().param_groups)['lr']
            except:
                current_lr = self.get_optimizer().param_groups[0]['lr']
            self._logger.info("[Epoch %04d] EpochLoss: %s LR: %s}"
                              %(i,
                                f'{epoch_loss:.010f}' if epoch_loss is not None else 'None',
                                f'{current_lr:.010f}' if current_lr is not None else 'None',))

    @abstractmethod
    def validation(self, *args, **kwargs):
        """
        This is called after each epoch.
        """
        raise NotImplementedError("Validation is not implemented in this solver.")

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
                self._logger.error("Unexpected error in type convertion of solver")
                self._logger.exception(e)

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

    @abstractmethod
    def _epoch_prehook(self, *args, **kwargs):
        pass

    def _epoch_callback(self, *args, **kwargs):
        """
        Default callback after `solver_epoch` is done.
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
        except AttributeError as e:
            self._logger.warning(f"Key absent in config: ({section},{key})")
            self._logger.exception(e)
            return default
        except Exception as e:
            self._logger.error(f"Unexpected error when reading params with key: ({section}, {key})")
            self._logger.exception(e)
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
