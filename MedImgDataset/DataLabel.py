import torch
from torch.utils.data import Dataset
import pandas as pd

class DataLabel(Dataset):
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


    def set_target_column(self, target):
        assert target in self._data_table.columns
        self._target_column = target

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

    def size(self, item):
        return self.__len__()

    def write(self, out_fname):
        self._data_table.to_csv(out_fname)


    def __len__(self):
        return len(self._data_table)

    def __getitem__(self, item):
        if self._target_column is None:
            return torch.tensor(self._data_table.iloc[item])
        else:
            return torch.tensor(self._data_table[self._target_column][item])

    def __str__(self):
        return self._data_table.to_string()
