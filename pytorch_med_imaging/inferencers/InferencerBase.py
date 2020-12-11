import os
from abc import abstractmethod
from pytorch_med_imaging.logger import Logger
import torch

class InferencerBase(object):
    """
    This is the base class of all inferencer, the inferencer needs to be configured first before use.
    The configueration is done through passing the dictionary.

    Parameters keys are:
        'indataset' (Dataset):
        'net' (nn.Module):
        'netstatedict' (torch.statedict):
        'batchsize' (int):
        'iscuda' (bool):


    Attributes
        _logger (logger.Logger):
            Use this logger to log anything or print anything.
    """
    def __init__(self, inferencer_configs):
        super(InferencerBase, self).__init__()

        # required
        self._in_dataset        = inferencer_configs['indataset']
        self._net               = inferencer_configs['net'] # note that this should be callable
        self._net_state_dict    = inferencer_configs['netstatedict']
        self._batchsize         = inferencer_configs['batchsize']
        self._iscuda            = inferencer_configs['iscuda']
        self._outdir            = inferencer_configs['outdir']
        assert os.path.isfile(self._net_state_dict), "Cannot open network checkpoint!"

        # optional
        self._logger = inferencer_configs.get('Logger', None)
        if self._logger is None:
            self._logger = Logger[self.__class__.__name__]

        assert isinstance(self._logger, Logger) or self._logger is None, "Incorrect logger."

        if 'target_data' in inferencer_configs:
            self._target_dataset = inferencer_configs['target_data']
            self._TARGET_DATASET_EXIST_FLAG = True
        else:
            self._TARGET_DATASET_EXIST_FLAG = False

        self._input_check()
        self._create_net()
        self._create_dataloader()


    def _match_type_with_network(self, tensor):
        """
        Return a tensor with the same type as the first weight of `self._net`. This function seems to cause CUDA
        error in pytorch 1.3.0

        Args:
            tensor (torch.Tensor or list): Input `torch.Tensor` or list of `torch.Tensor`

        Returns:
            out (torch.Tensor)
        """
        assert isinstance(tensor, list) or torch.is_tensor(tensor), "_match_type_with_network: input type error!"

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
        out = [ss.type(self._net_weight_type) for ss in tensor] if isinstance(tensor, list) else \
            tensor.type(self._net_weight_type)
        return out


    def get_net(self):
        return self._net

    @abstractmethod
    def _input_check(self):
        raise NotImplementedError

    @abstractmethod
    def _create_net(self):
        raise NotImplementedError

    @abstractmethod
    def _create_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def display_summary(self):
        raise NotImplementedError