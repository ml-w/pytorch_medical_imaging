from logger import Logger
from torch.utils.data import Dataset
from abc import *

class PMIDataBase(Dataset):
    def __init__(self, *args, **kwargs):
        self._logger = Logger[self.__class__.__name__]
        self._logger._verbose = Logger.global_logger._verbose

        super(PMIDataBase, self).__init__()

    def log_print_tqdm(self, msg, level=Logger.INFO):
        self._logger.log_print_tqdm(msg, level)

    def log_print(self, msg, level=Logger.INFO):
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
    