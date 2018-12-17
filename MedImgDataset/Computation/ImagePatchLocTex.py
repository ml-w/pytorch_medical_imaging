import torch
import torch.nn as nn

from numpy import argmax, array
from numpy.linalg import norm
from .. import ImageDataSet, ImagePatchesLoader
from .LocalBinaryPattern import LBP

class ImagePatchLocTex(ImagePatchesLoader):
    def __init__(self, *args, **kwargs):
        super(ImagePatchLocTex, self).__init__(*args, **kwargs)

        assert isinstance(self._base_dataset, ImageDataSet), "Currently only supports ImageDataset"
        assert self._base_dataset._byslices >= 0, "Currently only support load by slices."

        temp = self._base_dataset.data.squeeze().numpy()
        for i in xrange(len(self._base_dataset)):
            temp[i] = LBP(temp[i])
        temp = torch.tensor(temp).unsqueeze(1)
        self._base_dataset.data = torch.cat([self._base_dataset.data, temp], dim=1)

    def _getpos(self, item):
        # locate slice number of the patch
        slice_index = item / len(self._patch_indexes)

        # locate item position
        n = argmax(self._base_dataset._itemindexes > slice_index)
        range = [self._base_dataset._itemindexes[n - 1], self._base_dataset._itemindexes[n]]
        loc = slice_index - self._base_dataset._itemindexes[n-1]
        pos = loc / float(range[1] - range[0]) - 0.5
        return pos

    def _calculate_patch_pos(self, item):
        pos = self._getpos(item)
        patch_index = item % len(self._patch_indexes)
        p = self._patch_indexes[patch_index]
        return p[0], p[1], pos

    def _get_center(self):
        c = array(self._base_dataset[0].shape[-2:]) / 2.
        return c[0], c[1], 0.

    def _calculate_patch_dist(self, item):
        return norm(array(self._calculate_patch_pos(item)) - array(self._get_center()))


    def __getitem__(self, item):
        out_0 = super(ImagePatchLocTex, self).__getitem__(item)
        if isinstance(item, slice):
            start = item.start if not item.start is None else 0
            stop = item.stop if not item.stop is None else self.__len__()
            step = item.step if not item.step is None else 1
            feats = []
            for i in xrange(start, stop, step):
                feats.append(torch.cat([torch.tensor(self._calculate_patch_pos(i)).float(),
                                        torch.tensor([self._calculate_patch_dist(i)]).float()]))
            return out_0.squeeze(), torch.stack(feats)
        else:
            return out_0



