from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from ..logger import Logger

__all__ = ['PMIBatchSamplerFactory']

class PMIBatchSamplerFactory(object):
    def __init__(self):
        self._logger = Logger[__class__.__name__]

    def produce_object(self, config):
        try:
            requested_sampler_type = config['LoaderParams']
            config.get
        except AttributeError:
            self._logger.exception("Reading attribute from config file failed.")
            self._logger.warning("Falling back to default dataloader.")
            return DataLoader
