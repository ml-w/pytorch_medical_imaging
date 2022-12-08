import torch
from mnts.mnts_logger import MNTSLogger
from torch.utils.data import Dataset
from abc import *
from typing import Iterable, Union, Any

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
    def __getitem__(self, item) -> torch.Tensor:
        """All classes that inherit this should have this function implemented to return a
        torch.Tensor instance."""
        raise NotImplementedError("Unfinished class implementation.")

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("Unfinished class implementation.")

    @abstractmethod
    def get_unique_IDs(self) -> Iterable[str]:
        r"""Obtain the IDs of the data.

        Returns:
            Iterable
        """
        raise NotImplemented("Unfinished class implementation")

    @abstractmethod
    def get_data_by_ID(self, item: Any) -> torch.Tensor:
        r"""Obtain the data using an identifier. In this package, the identifier is generally a string globbed from
        somewhere (e.g., the filename). Although it is adviced that these ID should be unique, it is not compulsory.

        Args:
            item (Any): Index.

        Returns:
            ``torch.Tensor``
        """
        raise NotImplemented("Unfinished class implementation!")

    def batch_done_callback(self, *args):
        raise NotImplementedError("Batch done callback was not implemented in this class")