import torch
import pandas as pd
from pathlib import Path
from .PMIDataBase import PMIDataBase
from .DataLabel import DataLabel
from typing import Optional, Union, Iterable, Type, Any

class DataLabelConcat(DataLabel):
    def __init__(self,
                 data_table: str,
                 dtype: Optional[type] = str,
                 config: Optional[dict] = {}):
        """
        Datasheet should b arrange with rows of values

        Args:
            data_table (str):
                Directory to datatable
            dtype (type, Optional):
                Datatype for concat. Default to str.
            config (dict, Optional):
                Config for each dtype, see constructor for more. Default to {}.

        Attributes:
            _deliminator:
                For use when dtype == str, the character between the joined sequences.

        """
        super(DataLabelConcat, self).__init__(data_table)
        self.dtype = dtype

        self._deliminator = config.get('deliminator', ' ')
        self._reconstruct_data_table()

    def _reconstruct_data_table(self):
        r""""""
        _df = self._data_table.copy()
        rows = {}
        for key, row in _df.groupby(level=0):
            rows[key] = [self._concat(row[col]) for col in row]
        self._data_table = pd.DataFrame(data=rows.values(), index=rows.keys(), columns=_df.columns)
        self._data_table.index.set_names = _df.index.names

    def _concat(self, target) -> Any:
        r"""Type return is same as self.dtype"""
        if self.dtype == str:
            if len(target) > 1 and not isinstance(target, str):
                return self._deliminator.join([r.rstrip() for r in target])
            elif len(target) == 1 and isinstance(target, pd.Series):
                return target.item()
            else:
                return target
        elif self.dtype == int:
            return [int(o) for o in out]
        else:
            raise AttributeError(f"dtype is not supported, got {self.dtype}")

    def __len__(self):
        return len(self._data_table.index)

    def __getitem__(self, item):
        if self._get_table is None:
            self._get_table = self._data_table

        if isinstance(item, int):
            out = self._get_table.iloc[item]
        else:
            out = self._get_table.loc[item]

        if isinstance(out.index, pd.MultiIndex):
            out = self._concat(out)

        try:
            # if multiple rows are requested, a pandas dataframe object is directly returned
            return torch.tensor(out)
        except:
            return out

    def __str__(self):
        return self._data_table.to_string()

