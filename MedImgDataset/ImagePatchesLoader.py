from torch import cat, stack
from torch.utils.data import Dataset

class ImagePatchesLoader(Dataset):
    def __init__(self, base_dataset, patch_size, patch_stride, include_last_patch=True,
                 axis=None, reference_dataset=None):
        super(ImagePatchesLoader, self).__init__()

        assert axis is None or len(axis) == 2, "Axis argument should contain the two axises that forms the base image."
        assert isinstance(base_dataset, Dataset)
        assert reference_dataset is None or isinstance(reference_dataset, ImagePatchesLoader)

        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]
        if isinstance(patch_stride, int):
            patch_stride = [patch_stride, patch_stride]

        self._base_dataset = base_dataset
        self._patch_size = patch_size
        self._patch_stride = patch_stride
        self._axis = axis if not axis is None else [0, 1]
        self._include_last_patch= include_last_patch
        self._patch_indexes = []      # [(xmin, ymin), ... ], patches has

        # check axis
        assert len(self._patch_size) <= 2, "Support 2D patches only."
        assert len(self._patch_stride) <= 2, "Support 2D patches only."
        assert self._base_dataset[0].dim() <= len(self._axis) + 1, "Wrong dimension settings."
        self._unit_dimension = self._base_dataset[0].size()
        self._slice_dim = self._base_dataset[0].dim()

        if reference_dataset is None:
            self._calculate_patch_indexes()
            pass
        else:
            self._patch_indexes = reference_dataset._patch_indexes

    def _calculate_corner_range(self):
        pass

    def _calculate_patch_indexes(self):
        X, Y = self._axis

        Xlen, Ylen = self._unit_dimension[X], self._unit_dimension[Y]
        Xpat, Ypat = self._patch_size
        Xstr, Ystr = self._patch_stride

        nX = (Xlen - Xpat) / Xstr
        nY = (Ylen - Ypat) / Ystr

        resX = (Xlen - Xpat) % Xstr
        resY = (Ylen - Ypat) % Ystr

        func = lambda x, y: (x * Xstr, y * Ystr)

        for i in xrange(nX + 1):
            for j in xrange(nY + 1):
                self._patch_indexes.append(func(i, j))

                if resY != 0 and j == nY and self._include_last_patch:
                    self._patch_indexes.append([i * Xstr, j * Ystr + resY])
            if resX != 0 and i == nX and self._include_last_patch:
                for k in xrange(nY + 1):
                    self._patch_indexes.append([i * Xstr + resX, k * Ystr])
            if resX != 0 and resY != 0 and i == nX and j == nY and self._include_last_patch:
                self._patch_indexes.append([i * Xstr + resX, j * Ystr + resY])


    def __len__(self):
        return len(self._patch_indexes) * len(self._base_dataset)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start if not item.start is None else 0
            stop = item.stop if not item.stop is None else self.__len__()
            step = item.step if not item.step is None else 1
            return stack([self.__getitem__(i) for i in xrange(start, stop, step)], 0)
        else:
            slice_index = item / len(self._patch_indexes)
            patch_index = item % len(self._patch_indexes)

            p = self._patch_indexes[patch_index]
            s = self._base_dataset[slice_index]

            indexes = []
            for i in xrange(self._slice_dim):
                if i == self._axis[0]:
                    indexes.append(slice(p[0], p[0] + self._patch_size[0]))
                elif i == self._axis[1]:
                    indexes.append(slice(p[1], p[1] + self._patch_size[1]))
                else:
                    indexes.append(slice(None))
            return s[indexes]

