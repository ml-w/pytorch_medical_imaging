from MedImgDataset import ImageDataSet
from torch.utils.data import Dataset
from torch import cat
import numpy as np


class ImageDataSetMultiChannel(Dataset):
    def __init__(self, *args):
        assert np.all([isinstance(dat, ImageDataSet) for dat in args]), \
                "All input must be children of ImageDataSet"
        assert np.all([dat.size() == args[0].size() for dat in args]), \
                "Sizes of all input must be the same"

        self._basedata = args

        # Recalculate size
        c = np.sum([dat.size()[1] for dat in self._basedata])
        s = list(self._basedata[0].size())
        s[1] = c
        self._size = tuple(s)

    def size(self, int=None):
        return self._size

    def __getitem__(self, item):
        return cat([dat[item] for dat in self._basedata])