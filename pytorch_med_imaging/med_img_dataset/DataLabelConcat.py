import torch
import pandas as pd
from pathlib import Path
from .PMIDataBase import PMIDataBase
from .DataLabel import DataLabel
from typing import Optional, Union, Iterable, Type, Any, Tuple

class DataLabelConcat(DataLabel):
    r"""Class to concatenate data labels from a specified data table.

    This class processes a data table where each row may contain multiple values
    that need to be combined into a single value based on the specified data type.
    It allows for flexible configuration of the concatenation process, including
    the choice of delimiter for string concatenation.

    .. warning::
        Currently, this class is written for text inputs. If the content are text,
        they are concat into the same sentence. Otherwise, if the data type are
        numbers, a list would be returned to keep the length flexible.

    Args:
        data_table (str):
            The path to the data table file containing the labels.
        dtype (type, Optional):
            The data type for the concatenated result. Default is str, but can be
            set to int or other types as needed.
        config (dict, Optional):
            A dictionary of configuration options, including settings specific to
            each data type. Default is an empty dictionary.

    Attributes:
        _deliminator (str):
            The character used to separate concatenated strings when dtype is str.

    Examples:
    ---------
        Input:
            +----+-----------------------------+
            | ID | Text Value                  |
            +====+=============================+
            | A  | Some text written           |
            +----+-----------------------------+
            | A  | Another text sentence       |
            +----+-----------------------------+
            | B  | First line of text B        |
            +----+-----------------------------+
            | B  | Second line of text B       |
            +----+-----------------------------+
            | B  | Third line of text C        |
            +----+-----------------------------+

        >>> d = DataLabelConcat(input)
        >>> d['A']
        # "Some text written Another text sentence"
        >>> d['B']
        # "First line of text B Second line of text B Third line of text C"


    """
    def __init__(self,
                 data_table: str,
                 dtype: Optional[type] = str,
                 config: Optional[dict] = {}):
        super(DataLabelConcat, self).__init__(data_table)
        self._dtype = dtype
        self._deliminator = config.get('deliminator', ' ')
        self._reconstruct_data_table()

    def _reconstruct_data_table(self):
        r""""""
        _df = self._data.copy()
        rows = {}
        for key, row in _df.groupby(level=0):
            rows[key] = [self._concat(row[col]) for col in row]
        self._data = pd.DataFrame(data=rows.values(), index=rows.keys(), columns=_df.columns)
        self._data.index.set_names = _df.index.names

    def _concat(self, target) -> Any:
        r"""Type return is same as self.dtype"""
        if self._dtype == str:
            if len(target) > 1 and not isinstance(target, str):
                return self._deliminator.join([r.rstrip() for r in target])
            elif len(target) == 1 and isinstance(target, pd.Series):
                return target.item()
            else:
                return target
        elif self._dtype == int:
            return [int(o) for o in out]
        else:
            raise AttributeError(f"dtype is not supported, got {self.dtype}")

    def __getitem__(self, item: Union[int, slice, str]) -> Union[Any, Tuple[Any]]:
        r"""Returns an item from the data table.

        Args:
            item (int, slice, str):
                Unique ID or numeric index

        Returns:
            torch.Tensor or np.ndarray or str:
                The concatenated product.
        """
        if isinstance(item, (int, slice)):
            out = self._data.iloc[item]
        else:
            out = self._data.loc[item]

        if len(out) == 1 and not isinstance(out, str):
            return out.item()
        elif self._dtype == str:
            # If string, flattens the array
            return out.values.flatten()
        else:
            # Otherwise, return as an array
            return out.values

        try:
            return torch.tensor(out)
        except:
            return out
