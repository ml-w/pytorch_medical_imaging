import os
from abc import abstractmethod
from ..logger import Logger
from ..networks import *

import torch
import torchio as tio
import ast

class InferencerBase(object):
    """
    This is the base class of all inferencer, the inferencer needs to be configured first before use.
    The configueration is done through passing the dictionary. This class observes the SolverParams
    section of the ini file.

    Args:
        inferencer_configs (dict):
            Basic configuration that constructs an inferencer.

    Attributes
        _logger (logger.Logger):
            Use this logger to log anything or print anything.
    """
    def __init__(self, inferencer_configs, **kwargs):
        super(InferencerBase, self).__init__()

        # these attributes require evaluation and cannot be loaded directly from the ini file.
        default_attr = {
            'batch_size':       None,
            'net':              None,
            'net_state_dict':   None,
            'iscuda':           None,
            'outdir':           None,
            'pmi_data_loader':  None
        }
        default_attr.update((k, inferencer_configs[k]) for k in default_attr.keys() & inferencer_configs.keys())
        required_att = ('net', 'net_state_dict', 'iscuda', 'outdir', 'pmi_data_loader')

        self._logger = inferencer_configs.get('Logger', None)
        if self._logger is None:
            self._logger = Logger[self.__class__.__name__]

        if any([default_attr[k] is None for k in required_att]):
            self._logger.error("Some required attributes are not specified.")
            raise AttributeError(f"Must specify these attributes: {','.join(required_att)}")
        self.__dict__.update(default_attr)

        self._config = kwargs.get('config', None)

        # optional attributes
        default_attr = {
            'unpack_keys_inf': ['input'],
        }
        self._load_default_attr(default_attr)


        assert isinstance(self._logger, Logger) or self._logger is None, "Incorrect logger."

        if 'target_data' in inferencer_configs:
            self._target_dataset = inferencer_configs['target_data']
            self._TARGET_DATASET_EXIST_FLAG = True
        else:
            self._TARGET_DATASET_EXIST_FLAG = False

        self._input_check()
        self._create_net()
        self._prepare_data()

        if  len(kwargs):
            self._logger.warning("Some inferencer configs were not used: {}".format(kwargs))

    def _load_default_attr(self, default_dict):
        r"""
        Load default dictionary as attr from ini config, [SolverParams] section. Generally called from the
        children class when you want to add something to self.__dict__ from the ini file conveniently.
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

    def _match_type_with_network(self, tensor):
        """
        Return a tensor with the same type as the first weight of `self._net`. This function seems to cause CUDA
        error in pytorch 1.3.0

        Args:
            tensor (torch.Tensor or list): Input `torch.Tensor` or list of `torch.Tensor`

        Returns:
            out (torch.Tensor)
        """
        assert isinstance(tensor, list) or torch.is_tensor(tensor) or isinstance(tensor, tuple),\
            "_match_type_with_network: input type error! Expected list, tuple or torch.Tensor, "\
            "got {} instead.".format(type(tensor))

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
            self._logger.warning("Can't determine if type is already followed.")

        # We also expect list or tuple input too.
        out = [ss.type(self._net_weight_type) for ss in tensor] if isinstance(tensor, list) or \
                                                                   isinstance (tensor,tuple) else \
            tensor.type(self._net_weight_type)
        return out

    def get_net(self):
        return self.net

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

    def _input_check(self):
        assert os.path.isfile(self.net_state_dict), f"Cannot open network checkpoint at {self.net_state_dict}"

    def _create_net(self):
        r"""Try to create network and load state dict"""
        if not hasattr(self.net, 'forward'):
            self._logger.info("Creating network...")
            self.net = ast.literal_eval(self.net)
            if not hasattr(self.net, 'forward'):
                raise AttributeError("Cannot create network properly.")

        # Load state dict
        if isinstance(self.net, torch.nn.Module):
            self._logger.info(f"Loading network states from checkpoint: {self.net_state_dict}")
            self.net.load_state_dict(torch.load(self.net_state_dict))

        self.net = self.net.eval()

        # Move to GPU
        if self.iscuda:
            self.net = self.net.cuda()

    @abstractmethod
    def _prepare_data(self):
        raise NotImplementedError

    @abstractmethod
    def display_summary(self):
        raise NotImplementedError

    def _unpack_minibatch(self, minibatch, unpacking_keys):
        r"""Unpack mini-batch drawn by torchio.Queue or torchio.SubjectsDataset.
        TODO: allow custom modification after unpacking, e.g. concatentation
        """
        out = []
        for key in unpacking_keys:
            try:
                out.append(minibatch[key][tio.DATA])
            except AttributeError:
                out.append(minibatch[key])
        return out