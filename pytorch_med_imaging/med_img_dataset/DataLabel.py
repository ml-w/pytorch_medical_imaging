import torch
import pandas as pd
import numpy as np
from typing import Union, List, Any, Iterable, Union
from pathlib import Path
from .PMIDataBase import PMIDataBase

class DataLabel(PMIDataBase):
    """
    DataLabel is a class for managing labeled data from various sources such as CSV and Excel files.

    This class inherits from `PMIDataBase` and provides functionalities to load data, set target columns,
    compute new columns, and convert data to tensors or NumPy arrays.

    Attributes:
        _unique_ids (pd.Index):
            Unique identifiers for the rows in the dataset.
        _original_table (pd.DataFrame):
            Original data loaded from the input source.
        _data (pd.DataFrame):
            Processed data for analysis.
        _target_column (Union[str, List[str]]):
            Target column(s) for prediction or analysis.

    Args:
        data_table (Union[str, Path, pd.DataFrame]):
            Path to the data file or a DataFrame containing the data.
        **kwargs:
            Additional arguments passed to pandas read functions.

    Raises:
        AssertionError: If the input data_table is not a valid DataFrame.
        KeyError: If specified target column(s) are not found in the original data.
        Exception: If there is an issue converting data to a tensor.
    """
    def __init__(self, data_table, **kwargs):
        """
        Datasheet should b arrange with rows of values
        """
        super(DataLabel, self).__init__()
        if isinstance(data_table, (str, Path)):
            _p = Path(data_table)
            if _p.suffix == '.csv':
                data_table = pd.read_csv(str(data_table), index_col=[0], header=[0], **kwargs)
            elif _p.suffix == '.xlsx':
                data_table = pd.read_excel(str(data_table), index_col=[0], header=[0], **kwargs)
        assert isinstance(data_table, pd.DataFrame)
        if not data_table.index.is_unique:
            data_table


        # Convert to tensor
        self._unique_ids = data_table.index
        if not self._unique_ids.unique:
            print("Warning! Unique ID is not unique!")
        self._original_table: pd.DataFrame = data_table.copy()
        self._data          : pd.DataFrame = data_table.copy()  # Note that in base class this is pd.Series
        self._target_column : str          = None

    def set_computed_column(self, func, name='computed') -> int:
        r"""Set a column computed from other columns of the original data table.

        This method applies a given function to each row of the original data
        table and adds the resulting values as a new column. The name of the
        computed column can be specified. If the specified name already exists
        in the original table, a warning will be logged.

        Args:
            func (callable): A function that takes a row of the DataFrame and
                returns a computed value.
            name (str, optional): The name of the computed column. Defaults
                to 'computed'.

        Returns:
            int:
                - 0 if the operation was successful.
                - 1 if the provided function is not callable.

        Raises:
            Warning: If the computed column name already exists in the original
                     table.

        .. note::
            The computed column will be added to the `_original_table` DataFrame,
            and the target column will be set to the name of the computed column.
        """
        # ... existing code ...
        if not callable(func):
            self._logger.error("Input function {} is not callable.".format(func))
            return 1

        if name in self._original_table.columns:
            self._logger.warning(f"Name of computed column already exist in original table. This will break the"
                                 f"originality of the table")

        self._original_table[name] = self._original_table.apply(func, axis=1)
        self.set_target_column(name)
        return 0

    def set_target_column(self, target, dtype: type = None) -> int:
        r"""Sets the target column(s) for the DataLabel instance.

        This method allows you to specify the target column(s) for prediction or analysis.
        It can handle a single column as a string, multiple columns as a comma-separated string,
        or a list/tuple of column names. The method also supports type casting of the target columns.

        Args:
            target (Union[str, list, tuple]):
                The target column(s) to set.
                    - If a string is provided without commas, it sets a single target column.
                    - If a string with commas is provided, it sets multiple target columns.
                    - If a list or tuple is provided, it sets multiple target columns as well.
            dtype (type, optional):
                The data type to which the target column(s) should be cast.
                If not specified, no casting is performed.

        Raises:
            KeyError: If any specified target column(s) are not found in the original data table.

        Returns:
            int:
                - 0 if the operation was successful.
        """
        if isinstance(target, str):
            if not ',' in target:
                self._target_column = [target]
            else:
                target = []
                self._logger.debug("Multiple columns specified.")
                for t in target.split(','):
                    if not t in self._original_table.columns:
                        msg = ("Cannot found specified target column {} in data table! "
                               "Available columns are {}").format(t, self._data.columns)
                        self._logger.error(msg)
                        raise KeyError(msg)
                    else:
                        self._target_column.append(t)
        elif isinstance(target, (list, tuple)):
            if not all(t in self._original_table.columns for t in target):
                msg = ("Cannot found specified target column {} in data table! "
                       "Available columns are {}").format(t, self._data.columns)
                self._logger.error(msg)
                raise KeyError(msg)
            self._target_column = list(target)
        else:
            if not target in self._original_table.columns:
                msg = ("Cannot found specified target column {} in data table! "
                       "Available columns are {}").format(t, self._data.columns)
                self._logger.error(msg)
                raise KeyError(msg)
            self._target_column = target
        self._logger.debug("Setting columns to: {}".format(self._target_column))
        self._data = self._original_table[self._target_column].copy()

        # type cast
        if not dtype is None:
            try:
                if isinstance(dtype, (list, tuple)):
                    assert len(dtype) == len(self._target_column), "Dtype length must be the same as target col"
                    for t, d in zip(self._target_column, dtype):
                        self._data.loc[:, t] = self._data[t].astype(d)
                else:
                    self._data.loc[:, self._target_column] = self._data[self._target_column].astype(dtype)
            except Exception as e:
                self._logger.warning(f"Cannot cast column {self._target_column} to type {dtype}")
                self._logger.exception(e)
        return 0

    @staticmethod
    def from_csv(fname: str, **kwargs):
        r"""Creates a DataLabel instance from a CSV file.

        This static method reads data from the specified CSV file, converts the index to a string,
        and initializes a DataLabel object with the resulting DataFrame. It also prints the
        DataLabel instance for verification.

        Args:
            fname (str): The path to the CSV file to be read.
            **kwargs: Additional keyword arguments passed to `pd.read_csv`.

        Returns:
            DataLabel: An instance of the DataLabel class containing the data from the CSV file.

        Example:
            >>> datalabel = DataLabel.from_csv('data.csv')
            >>> print(datalabel)
        """
        df = pd.read_csv(fname, **kwargs, index_col=0)
        df.index = df.index.astype('str')
        datalabel = DataLabel(df)
        print(datalabel)
        return datalabel

    @staticmethod
    def from_xlsx(fname: str, sheet_name=None, header_row=False):
        r"""Creates a DataLabel instance from an Excel file.

        This static method reads data from the specified Excel file and initializes a DataLabel
        object with the resulting DataFrame. It allows you to specify which sheet to read from
        and handles the index conversion to string.

        Args:
            fname (str): The path to the Excel file to be read.
            sheet_name (str, optional): The name of the sheet to read. If None, the first sheet is used.
            header_row (bool, optional): Indicates whether to treat the first row as header. Defaults to False.

        Returns:
            DataLabel: An instance of the DataLabel class containing the data from the Excel file.

        Examples:
            >>> datalabel = DataLabel.from_xlsx('data.xlsx', sheet_name='Sheet1')
            >>> datalabel = DataLabel.from_xlsx('data.xlsx')
        """
        # Unique IDs should be recorded in
        xfile = pd.ExcelFile(fname)

        # Use first sheet if sheet_name is not specified
        if sheet_name is None:
            sheet_name = xfile.sheet_names[0]

        df = pd.read_excel(xfile, sheet_name, index_col=0)
        df.index = df.index.astype('str')
        datalabel = DataLabel(df)
        return datalabel

    @staticmethod
    def from_dict(in_dict: dict, **kwargs):
        df = pd.DataFrame.from_dict(in_dict, **kwargs)
        df.set_index(df.keys()[0])

        datalabel = DataLabel(df)
        return datalabel

    def map_to_data(self, target: PMIDataBase) -> int:
        r"""Maps target IDs to the original data table and updates the data.

        This method retrieves unique IDs from the provided target and attempts to map them
        to the original data table. If successful, it updates the internal data attribute
        with the rows corresponding to the target IDs. If a target column is specified,
        it further filters the data to include only that column.

        Args:
            target (PMIDataBase): An object that provides the method `get_unique_IDs()`
                to retrieve unique IDs for mapping.

        Returns:
            int:
                - 0 if the mapping was successful.
                - 1 if there was an error during the mapping process.

        Raises:
            Exception: Logs an error if the mapping process fails.
        """
        target_ids = target.get_unique_IDs()
        try:
            self._data = self._original_table.loc[target_ids]
            if not self._target_column is None:
                self._data = self._data[self._target_column]
            return 0
        except:
            self._logger.exception("Error when trying to map table to data.")
            return 1

    def get_unique_values(self) -> List[Any]:
        return list(self._data[self._target_column].unique())

    def size(self, item=None) -> int:
        return self.__len__()

    def write(self, out_fname: str) -> None:
        self._data.to_csv(out_fname)

    def to_numpy(self) -> np.ndarray:
        return self._data.to_numpy()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item) -> Union[torch.Tensor, np.ndarray]:
        r"""Returns an item from the datatable, as either torch.Tensor or
        np.ndarray.

        Args:
            item (str or int):
                Unique ID or numeric index

        Returns:
            torch.Tensor or np.ndarray:
                Output will be converted to either torch.Tensor or np.ndarray with
                `ndim = 2` and shaped :math:`(B \times C)`
        """
        out = super().__getitem__(item)
        if len(out) == 1:
            out = out.item()
        else:
            out: np.ndarray = out.to_numpy()
            if out.ndim == 1:
                out = out.reshape(1, 2)

        try:
            out = torch.tensor(out)
            if out.dim() == 0:
                out = out.reshape(1, -1)
            return torch.tensor(out)
        except TypeError:
            self._logger.warning(f"Output is a vector of multiple types! Returning as is.", no_repeat=True)
            return out
        except Exception as e:
            self._logger.info(f"Failed to convert to tensor {out}")
            self._logger.debug(f"Original error: {e}")
            return out

    def __str__(self):
        return self._data.to_string()
