from mnts.mnts_logger import MNTSLogger
from torch.utils.data import Dataset
from abc import *

class PMIDataBase(Dataset):
    def __init__(self, *args, **kwargs):
        self._logger = MNTSLogger[self.__class__.__name__]
        self._logger._verbose = MNTSLogger.global_logger._verbose

        super(PMIDataBase, self).__init__()

    def log_print_tqdm(self, msg, level=MNTSLogger.INFO):
        self._logger.log_print_tqdm(msg, level)

    def log_print(self, msg, level=MNTSLogger.INFO):
        self._logger.log_print(msg, level)


    @abstractmethod
    def size(self, i=None):
        raise NotImplementedError("Unfinished class implementation.")


    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError("Unfinished class implementation.")

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("Unfinished class implementation.")

    @abstractmethod
    def get_unique_IDs(self):
        raise NotImplemented("Unfnished class implementation")

    def apply_hook(self, func):
        r"""
        Apply this function to all output before it is returned.

        Args:
            func (callable):
                A function that returns a tensor or a list of tensors.
        """
        assert callable(func)
        self._get_item_hook = func


    def batch_done_callback(self, *args):
        raise NotImplementedError("Batch done callback was not implemented in this class")