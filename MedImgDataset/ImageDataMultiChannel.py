from .ImageData import ImageDataSet
from torch.utils.data import Dataset
from torch import cat, stack
import numpy as np


class ImageDataSetMultiChannel(Dataset):
    def __init__(self, *args):
        assert np.all([isinstance(dat, ImageDataSet) for dat in args]), \
                "All input must be children of ImageDataSet"
        assert np.all([dat.size() == args[0].size() for dat in args]), \
                "Sizes of all input must be the same"

        self._basedata = args
        self.data = self._basedata[0]       # For competibility

        # Inherit some of the properties of the inputs
        self._itemindexes = args[0]._itemindexes
        self._byslices = args[0]._byslices

        # Recalculate size
        c = np.sum([dat.size()[1] for dat in self._basedata])
        s = list(self._basedata[0].size())
        s[1] = c
        self._size = tuple(s)

        # Check if dtype are the same
        self._UNIQUE_DTYPE = np.all([dat.type() == self._basedata[0].type() for dat in self._basedata])

    def size(self, int=None):
        return self._size

    def Write(self, *args):
        try:
            self._basedata[0].Write(*args)
        except Exception as e:
            print(e)
            raise NotImplementedError("Base data have no Write() method")

    def __len__(self):
        return self.size()[0]

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = [item.start ,item.stop, item.step]
            start = start if not start is None else 0
            stop = stop if not stop is None else len(self._basedataset)
            step = step if not step is None else 1

            return stack([self.__getitem__(i) for i in range(start, stop, step)])
        else:
            # output datatype will follow the dtype of the first constructor argument
            if self._UNIQUE_DTYPE:
                return cat([dat[item] for dat in self._basedata])
            else:
                return cat([dat[item].type_as(self._basedata[0][item]) for dat in self._basedata])