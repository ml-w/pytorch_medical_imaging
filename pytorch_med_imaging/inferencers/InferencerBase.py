import os
from abc import abstractmethod
from mnts.mnts_logger import MNTSLogger
from ..networks import *
from ..pmi_data_loader.pmi_dataloader_base import PMIDataLoaderBase

import torch
import torchio as tio
import ast
from typing import Union, Any, Optional
from pathlib import Path

class InferencerBase(object):
    """
    This is the base class of all inferencer, the inferencer needs to be configured first before use.
    The configueration is done through passing the dictionary. This class observes the SolverParams
    section of the ini file.

    Args:
        inferencer_configs (dict):
            Basic configuration that constructs an inferencer.
        net (torch.nn.Module):
            The network.

    Attributes
        _logger (logger.Logger):
            Use this logger to log anything or print anything.
    """
    def __init__(self,
                 net: torch.nn.Module,
                 output_dir: Union[str, Path],
                 hyperparam_dict: dict,
                 use_cuda: bool,
                 pmi_data_loader: PMIDataLoaderBase,
                 debug: Optional[bool] = False, **kwargs):
        super(InferencerBase, self).__init__()
        self._logger = MNTSLogger[self.__class__.__name__]
        if not isinstance(net, torch.nn.Module):
            msg += f"Expect input net is an instance of nn.Module, but got type {type(net)} input."
            raise TypeError(msg)
        if not isinstance(hyperparam_dict, dict):
            msg += f"Expect hyperparam_dict to be a dictionary, but got type {type(hyperparam_dict)} input."
            raise TypeError(msg)

        self.net = net
        self.iscuda = use_cuda
        self.output_dir = output_dir
        self.hyperparam_dict = hyperparam_dict
        self.pmi_data_loader = pmi_data_loader
        self.debug = debug
        self.__dict__.update(hyperparam_dict)

        self.required_att = [
            'solverparams_unpack_keys_inference'
        ]
        self.check_attr()
        self._input_check()

        if self.iscuda:
            self._logger.info("Moving network to GPU.")
            self.net = self.net.cuda()

        if  len(kwargs):
            self._logger.warning("Some inferencer configs were not used: {}".format(kwargs))

    def check_attr(self) -> None:
        r"""Inherit this to add to the list of required_att."""
        if any([hasattr(self, k) is False for k in self.required_att]):
            missing = ', '.join([a for a in self.required_att if hasattr(self, a) is False])
            msg = f"The following attribute(s) is/are required to be specified in the SolverParams section but " \
                  f"is/are missing: {missing}"
            raise AttributeError(msg)

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

    def _match_type_with_network(self, tensor):
        """
        Return a tensor with the same type as the first weight of `self._net`. This function seems to cause CUDA
        error in pytorch 1.3.0. This will automatically move tensors to CUDA if self.net is already in GPU.

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
            if isinstance(tensor, (list, tuple)):
                if all([t.type() == self._net_weight_type for t in tensor]):
                    return tensor
            else:
                if tensor.type() == self._net_weight_type:
                    return tensor
        except Exception as e:
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

    def get_net(self):
        return self.net

    def _input_check(self):
        assert os.path.isfile(self.net_state_dict), f"Cannot open network checkpoint at {self.net_state_dict}"

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        r"""Load the checkpoint states"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            msg = f"Cannot open checkpoint at: {str(checkpoint_path)}!"
            raise IOError(msg)

        self._logger.info("Loading checkpoint " + str(checkpoint_path))
        self.get_net().load_state_dict(torch.load(str(checkpoint_path)), strict=False)

    @abstractmethod
    def _prepare_data(self):
        raise NotImplementedError

    @abstractmethod
    def display_summary(self):
        raise NotImplementedError

    @abstractmethod
    def write_out(self):
        raise NotImplementedError("This method must be implemented by the child class")

    def _unpack_minibatch(self, minibatch, unpacking_keys):
        r"""Unpack mini-batch drawn by torchio.Queue or torchio.SubjectsDataset.
        TODO: allow custom modification after unpacking, e.g. concatentation
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

    def set_dataloader(self, dataloader, data_loader_val=None):
        r"""These data_loaders are pytorch torch.utils.data.DataLoader objects, and they should be
        differentiated from the `pmi_data_loader`"""
        self._data_loader = dataloader
        self._data_loader_val = data_loader_val

    def set_pmi_data_loader(self, pmi_data_loader: PMIDataLoaderBase):
        # Do nothing, this is an entry point for overriding data loader by the child classes.
        self.pmi_data_loader = pmi_data_loader
