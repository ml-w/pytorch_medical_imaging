from torch import tensor
from numpy import argmax

from torch.utils.data import Dataset
from ..ImageData import ImageDataSet
from ..ImageDataMultiChannel import ImageDataMultiChannel

class ImageDataSetWithPos(Dataset):
    def __init__(self, imagedataset):
        super(ImageDataSetWithPos, self).__init__()

        assert isinstance(imagedataset, ImageDataSet) or isinstance(imagedataset, ImageDataMultiChannel)
        assert imagedataset._byslices >= 0, "Loading with position only supports by slice position currently."

        self._basedataset = imagedataset


    def size(self, int=None):
        if int is None:
            return self._basedataset.size()
        else:
            return self._basedataset.size()[int]

    def _getpos(self, item):
        # locate item position
        n = argmax(self._basedataset._itemindexes > item)
        range = [self._basedataset._itemindexes[n - 1], self._basedataset._itemindexes[n]]
        loc = item - self._basedataset._itemindexes[n-1]
        pos = loc / float(range[1] - range[0]) - 0.5                # anatomy normalized by ratio
        return pos


    def Write(self, *args):
        try:
            self._basedataset.Write(*args)
        except Exception as e:
            print(e)
            raise NotImplementedError("Base data have no Write() method")

    def __len__(self):
        return len(self._basedataset)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = [item.start ,item.stop, item.step]
            start = start if not start is None else 0
            stop = stop if not stop is None else len(self._basedataset)
            step = step if not step is None else 1
            pos = [self._getpos(i) for i in range(start, stop, step)]
            pos = tensor(pos).view(len(pos), 1, 1, 1)
        else:
            pos = tensor(self._getpos(item)).expand(1, 1, 1)
        return self._basedataset.__getitem__(item), pos