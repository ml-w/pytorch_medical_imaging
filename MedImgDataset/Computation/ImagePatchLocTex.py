import torch
import torch.nn as nn

import numpy as np
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
        n = np.argmax(self._base_dataset._itemindexes > slice_index)
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
        c = np.array(self._base_dataset[0].shape[-2:]) / 2.
        return c[0], c[1], 0.

    def _calculate_patch_dist(self, item):
        return np.linalg.norm(np.array(self._calculate_patch_pos(item)) - \
                              np.array(self._get_center()))


    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start if not item.start is None else 0
            stop = item.stop if not item.stop is None else self.__len__()
            step = item.step if not item.step is None else 1

            L = [self.__getitem__(i)[0] for i in xrange(start, stop, step)]
            out_0 = torch.stack([l[0] for l in L], 0)
            feats = torch.stack([l[1] for l in L], 0)
            return out_0.squeeze(), feats
        else:
            feats = torch.cat([torch.tensor(self._calculate_patch_pos(item)).float(),
                               torch.tensor([self._calculate_patch_dist(item)]).float()])
            return super(ImagePatchLocTex, self).__getitem__(item), feats
