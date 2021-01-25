import torch
import pandas as pd
from .PMIDataBase import PMIDataBase

class DataLabel(PMIDataBase):
    def __init__(self, data_table):
        """
        Datasheet should b arrange with rows of values
        """
        super(DataLabel, self).__init__()
        assert isinstance(data_table, pd.DataFrame)


        # Convert to tensor
        self._unique_ids = data_table.index
        if not self._unique_ids.unique:
            print("Warning! Unique ID is not unique!")
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

    def set_target_column(self, target):
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
        return 0

    @staticmethod
    def from_csv(fname, **kwargs):
        df = pd.read_csv(fname, **kwargs, index_col=0)
        df.index = df.index.astype('str')
        datalabel = DataLabel(df)
        return datalabel

    @staticmethod
    def from_xlsx(fname, sheet_name=None, header_row=False):
        # Unique IDs should be recorded in
        xfile = pd.ExcelFile(fname)
        df = pd.read_excel(xfile, sheet_name, index_col=0)
        df.index = df.index.astype('str')
        datalabel = DataLabel(df)
        return datalabel

    @staticmethod
    def from_dict(dict):
        df = pd.DataFrame.from_dict(dict)
        df.set_index(df.keys()[0])

        datalabel = DataLabel(df)
        return datalabel

    def map_to_data(self, target, target_id_globber=None):
        target_ids = target.get_unique_IDs(target_id_globber)
        try:
            self._data_table = self._original_table.loc[target_ids]
        except:
            import traceback as tr
            tr.print_last()

    def get_unique_values(self):
        return list(self._data_table[self._target_column].unique())

    def size(self, item=None):
        return self.__len__()

    def write(self, out_fname):
        self._data_table.to_csv(out_fname)

    def to_numpy(self):
        return self._data_table.to_numpy()

    def __len__(self):
        return len(self._data_table)

    def __getitem__(self, item):
        if isinstance(item, int):
            if self._target_column is None:
                out = self._data_table.iloc[item]
            else:
                out = self._data_table[self._target_column].iloc[item]
        elif isinstance(item, str):
            if self._target_column is None:
                out = self._data_table.loc[item]
            else:
                out = self._data_table[self._target_column].loc[item]
        else:
            out = self._data_table.loc[item]
        return torch.tensor(out)

    def __str__(self):
        return self._data_table.to_string()
