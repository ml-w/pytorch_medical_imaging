import torch
from mnts.mnts_logger import MNTSLogger
from torch.utils.data import Dataset
from abc import *
from typing import Iterable, Union, Any, List
import pandas as pd

from pytorch_med_imaging.utils.uid_ops import get_unique_IDs


class PMIDataBase(Dataset):
    r"""This is the base class of PMI Datasets.

    The key of this class is that it restrict images to have a standard

    Override Guide
    ^^^^^^^^^^^^^^

    Implement Abstract Methods:
        To implement your own data class, you would need to implement a few abstract methods including:

        1. :meth:`__getitem__`
        2. :attr:`data` [Property]
        3. :attr:`size` [Property]
        4. :attr:`dtype` [Property]

        These are required by the torch's default dataloader and ensure the data can be iterated by the
        official dataloader.

    Definitions of Attributes:
        By default, the data is held in :attr:`_data` which should only be accessed by :attr:`data`.
        The data should be stored as `pd.Series`, which index are the globbed unique IDs, and the name
        of the series is the subject data key (as in `torchio.Subject`).


    Attributes:
       _data (pd.DataFrame or pd.Series):
           This should be an ID mapper, which maps the UID to integer of individual data.
           It is structured as a pandas DataFrame or Series where the index represents unique
           identifiers for each data entry and the values represent the corresponding data.

       _logger (MNTSLogger):
           Logger instance for tracking the operations performed in this class. It helps in
           debugging and monitoring the dataset operations.

   """
    def __init__(self, *args, **kwargs):
        self._logger = MNTSLogger[self.__class__.__name__]
        super(PMIDataBase, self).__init__()
        self._data: pd.Series = None # This should map the uid to the data

    @property
    def id(self) -> List[str]:
        r"""Returns the IDs of the loaded data."""
        if len(self._data) == 0:
            self._logger.warning("Trying to get uid from empty dataset")
        return self._data.index.to_list()

    @property
    def data(self) -> pd.Series:
        r"""Returns the data.

        Returns:
            pd.Series: Note that the class could change in child class.
        """
        return self._data

    @abstractmethod
    def size(self, i=None):
        raise NotImplementedError("Unfinished class implementation.")

    @abstractmethod
    def __getitem__(self, item) -> torch.Tensor:
        """All classes that inherit this should have this function implemented to return a
        torch.Tensor instance."""
        if isinstance(item, str):
            # Get item with unique ID
            return self._data.loc[item]
        else:
            return self._data.iloc[item]

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("Unfinished class implementation.")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_unique_IDs(self) -> Iterable[str]:
        r"""Obtain the IDs of the data.

        Returns:
            Iterable
        """
        return self._data.index.unique().to_list()

    def get_data_by_ID(self, item: Any) -> torch.Tensor:
        r"""Obtain the data using an identifier. In this package, the identifier is generally a string globbed from
        somewhere (e.g., the filename). Although it is adviced that these ID should be unique, it is not compulsory.

        Args:
            item (Any): Index.

        Returns:
            ``torch.Tensor``
        """
        if isinstance(item, str):
            return self.__getitem__(item)
        else:
            raise KeyError("Unique ID of items must be string or integer.")

    def batch_done_callback(self, *args) -> None:
        r"""This will be called after each mini-batch if implemented."""
        raise NotImplementedError("Batch done callback was not implemented in this class")

    def remap_to_master_data(self, target):
        r"""Reorder the dataset to align with the target dataset.

        This function modifies the current dataset so that its records are reordered
        to match the order of the records in the target dataset.

        Args:
            target (PMIDataBase): The target dataset whose order is to be used for
            remapping the current dataset.

        Raises:
            ValueError: If the target dataset is not compatible with the current dataset.

        Example:
            >>> current_data.remap_to_master_data(target_data)

        """
        # Reorder the current dataset based on the target dataset's order
        original_ids = self.id.copy()
        self._data = self._data.loc[target.data.index.intersection(self._data.index)]
        self._logger.info("Data UIDs successfully remapped to align with target dataset.")
        self._logger.debug(f"Before: [{','.join(original_ids)}], After: [{','.join(self.id)}]")

    def sort_uid(self) -> None:
        r"""Sort the data based on uid.

        .. note::
            Original indexing will be lost! This will affect the data you get when you put interger for __getitem__
        """
        self._data.sort_index(inplace=True)

    @property
    def dtype(self) -> Any:
        r"""Returns the type of the data.

        Returns:
            Any: If the data is pandas
        """
        if isinstance(self._data, pd.Series):
            return self._data.dtype
        elif isinstance(self._data, pd.DataFrame):
            return self._data.dtypes
        else:
            msg = "Default implementation for pd.Series and pd.DataFrame only."
            raise AttributeError(msg)

    def remap_data_by_ids(self, id_list: Iterable):
        r"""


        Args:
            id_list (Iterable):
                This list must be unique

        Returns:

        """
        if not len(self.data):
            raise KeyError("There's no data to map")
        self._logger.debug("Remapping data.")
        self._data = self._data.loc[id_list]