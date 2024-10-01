import torch
import pandas as pd
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
        self._get_table = None
        self._data_table = data_table
        self._original_table = data_table
        self._target_column = None

    def set_computed_column(self, func, name='computed'):
        if not callable(func):
            self._logger.error("Input function {} is not callable.".format(func))
            return 1

        self._data_table[name] = self._data_table.apply(func, axis=1)
        self.set_target_column(name)
        return 0

    def set_target_column(self, target, dtype: type = None):
        if target.find(','):
            self._target_column = []
            self._logger.debug("Multiple columns specified.")
            for t in target.split(','):
                if not t in self._data_table.columns:
                    self._logger.warning("Cannot found specified target column {} in data table!"  
                                         "Available columns are {}".format(t, self._data_table.columns))
                else:
                    self._target_column.append(t)
        else:
            if not target in self._data_table.columns:
                self._logger.warning("Cannot found specified target column '{}' in data table!"
                                     "Available columns are {}".format(target, self._data_table.columns))
                self._logger.warning("Setting target to {} anyways.".format(target))
            self._target_column = target
        self._logger.debug("columns are: {}".format(self._target_column))
        self._get_table = self._data_table[self._target_column]

        # type cast
        if not dtype is None:
            try:
                self._get_table.loc[:, self._target_column] = self._get_table[self._target_column].astype(dtype)
            except Exception as e:
                self._logger.warning(f"Cannot cast column {self.target_column} to type {dtype}")
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

    def map_to_data(self, target, target_id_globber=None):
        target_ids = target.get_unique_IDs(target_id_globber)
        try:
            self._data_table = self._original_table.loc[target_ids]
            if not self._target_column is None:
                self._get_table = self._data_table[self._target_column]
            return 0
        except:
            self._logger.exception("Error when trying to map table to data.")
            return 1

    def get_unique_values(self):
        return list(self._data_table[self._target_column].unique())

    def size(self, item=None):
        return self.__len__()

    def write(self, out_fname):
        self._data_table.to_csv(out_fname)

    def to_numpy(self):
        return self._data_table.to_numpy()

    def get_unique_IDs(self, *args):
        return list(self._data_table.index)

    def get_data_by_ID(self, id):
        return self.__getitem__(id)

    def __len__(self):
        return len(self._data_table)

    def __getitem__(self, item):
        if self._get_table is None:
            self._get_table = self._data_table

        if isinstance(item, int):
            out = self._get_table.iloc[item]
        else:
            out = self._get_table.loc[item]
        if len(out) == 1:
                out = out.item()
        else:
                out = out.to_numpy()

        try:
            # if multiple rows are requested, a pandas dataframe object is directly returned
            return torch.tensor(out)
        except:
            self._logger.info(f"Failed to convert to tensor {out}")
            return out


    def __str__(self):
        return self._data_table.to_string()

