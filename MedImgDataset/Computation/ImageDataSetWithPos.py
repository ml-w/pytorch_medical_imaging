from torch import tensor
from numpy import argmax

from ..ImageData import ImageDataSet

class ImageDataSetWithPos(ImageDataSet):
    def __init__(self, *args, **kwargs):
        super(ImageDataSetWithPos, self).__init__(*args, **kwargs)
        assert self._byslices >= 0, "Loading with position only supports by slice position currently."

    def _getpos(self, item):
        # locate item position
        n = argmax(self._itemindexes > item)
        range = [self._itemindexes[n - 1], self._itemindexes[n]]
        loc = item - self._itemindexes[n-1]
        pos = loc / float(range[1] - range[0]) - 0.5                # anatomy normalized by ratio
        return pos

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = [item.start ,item.stop, item.step]
            start = start if not start is None else 0
            stop = stop if not stop is None else len(self.data)
            step = step if not step is None else 1
            pos = [self._getpos(i) for i in xrange(start, stop, step)]
            pos = tensor(pos).view(len(pos), 1, 1, 1)
        else:
            pos = tensor(self._getpos(item)).expand(1, 1, 1)
        return super(ImageDataSetWithPos, self).__getitem__(item), pos