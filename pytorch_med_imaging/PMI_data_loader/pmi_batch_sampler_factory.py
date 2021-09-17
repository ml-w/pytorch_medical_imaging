from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from .pmi_batch_zero_pad_sampler import *

from ..logger import Logger

__all__ = ['PMIBatchSamplerFactory']

class PMIBatchSamplerFactory(object):
    def __init__(self):
        self._logger = Logger[__class__.__name__]

    def produce_object(self, dataset, config, force_inference=None):
        try:
            requested_sampler_type = config['LoaderParams']['PMI_loader_name']
            sampler_kwargs = config['LoaderParams'].get('PMI_loader_kwargs', dict())
            param_batchsize = int(config['RunParams'].get('batch_size'))
            run_mode = config['General'].get('run_mode', 'training')
            run_mode = run_mode == 'test' or run_mode == 'testing' or run_mode == "inference"
            if force_inference is not None:
                if isinstance(force_inference, bool):
                    run_mode = force_inference

            self._logger.info(f"Creating loader {requested_sampler_type} in mode: "
                               f"{'inference' if run_mode else'train'}.")

            sampler_kwargs = eval(sampler_kwargs)
            if not isinstance(sampler_kwargs, dict):
                raise TypeError("Incorrect specification for PMI_loader_kwargs")


            product = eval(requested_sampler_type)(dataset,
                                                   batch_size=param_batchsize,
                                                   shuffle=not run_mode,  # if eval, don't shuffle
                                                   drop_last =not run_mode,  # if eval, don't drop last
                                                   **sampler_kwargs
                                                   )
            product.class_name = requested_sampler_type
            return product

        except AttributeError:
            self._logger.exception("Reading attribute from config file failed.")
            self._logger.warning("Falling back to default dataloader.")
            return DataLoader
