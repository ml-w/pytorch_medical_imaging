import torch
import pandas as pd
import numpy as np
from typing import Union
from pathlib import Path
from .PMIDataBase import PMIDataBase

class DataLabel(PMIDataBase):
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

    def set_computed_column(self, func, name='computed'):
        if not callable(func):
            self._logger.error("Input function {} is not callable.".format(func))
            return 1

        if name in self._original_table.columns:
            self._logger.warning(f"Name of computed column already exist in original table. This will break the"
                                 f"originality of the table")

        self._original_table[name] = self._original_table.apply(func, axis=1)
        self.set_target_column(name)
        return 0

    def set_target_column(self, target, dtype: type = None):
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
    def from_csv(fname, **kwargs):
        df = pd.read_csv(fname, **kwargs, index_col=0)
        df.index = df.index.astype('str')
        datalabel = DataLabel(df)
        print(datalabel)
        return datalabel

    @staticmethod
    def from_xlsx(fname, sheet_name=None, header_row=False):
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
    def from_dict(dict, **kwargs):
        df = pd.DataFrame.from_dict(dict, **kwargs)
        df.set_index(df.keys()[0])

        datalabel = DataLabel(df)
        return datalabel

    def map_to_data(self, target):
        target_ids = target.get_unique_IDs()
        try:
            self._data = self._original_table.loc[target_ids]
            if not self._target_column is None:
                self._data = self._data[self._target_column]
            return 0
        except:
            self._logger.exception("Error when trying to map table to data.")
            return 1

    def get_unique_values(self):
        return list(self._data[self._target_column].unique())

    def size(self, item=None):
        return self.__len__()

    def write(self, out_fname):
        self._data.to_csv(out_fname)

    def to_numpy(self):
        return self._data.to_numpy()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item) -> Union[torch.Tensor, np.ndarray]:
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
