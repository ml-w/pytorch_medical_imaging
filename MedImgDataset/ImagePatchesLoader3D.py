import torch
from torch import cat, stack, tensor, zeros
from torch.utils.data import Dataset
from ImageData import ImageDataSet
from ImageDataMultiChannel import ImageDataSetMultiChannel
from ImageDataAugment import ImageDataSetAugment
from tqdm import *
import numpy as np


class ImagePatchesLoader3D(Dataset):
    def __init__(self, base_dataset, patch_size, patch_stride=-1, include_last_patch=True,
                 reference_dataset=None, pre_shuffle=False, random_patches=-1,
                 random_from_distribution=None, renew_index=True):
        """ImagePatchesLoader(self, base_dataset, patch_size, patch_stride, include_last_patch=True,
                 axis=None, reference_dataset=None, pre_shuffle=False, random_patches=-1,
                 random_from_distribution=None) --> ImagePatchesLoader
        """
        super(ImagePatchesLoader3D, self).__init__()

        assert isinstance(base_dataset, ImageDataSet) or isinstance(base_dataset, ImageDataSetMultiChannel)
        assert reference_dataset is None or isinstance(reference_dataset, ImagePatchesLoader3D)
        assert not (patch_stride == -1 and random_patches == -1), \
            "You must select a patch stride if not using random patches."

        if isinstance(patch_size, int):
            patch_size = [patch_size] * 3
        if isinstance(patch_stride, int):
            patch_stride = [patch_stride] * 3

        self._pre_shuffle = pre_shuffle     # If pre-shuffle is True, patch indexes will be shuffled.
        self._base_dataset = base_dataset   # Base ImageDataSet.
        self._patch_size = patch_size       # Dimension of each patches.
        self._patch_stride = patch_stride   # Stide of patch extraction windows, ignored if random_patches is True.
        self._include_last_patch= include_last_patch
        self._patch_indexes = []            # [(xmin, ymin), ... ], corners of the patches.
        self._random_patches = False
        self._axis = [-3, -2, -1]           # Default to last three axises
        self._random_counter = 0            # This counter is used to re-calculate patch corner after each epoch
        self._random_from_distrib = \
            random_from_distribution        # This is a function to covert a region of an image to a
                                            # probability map
        self._renew_index = renew_index     # If this is true, index will be renewed after each epoch
        self.data = self._base_dataset.data

        # check axis
        self._unit_dimension = self._base_dataset[0].size()
        self._patch_ndim = self._base_dataset[0].dim()

        if reference_dataset is None:
            if random_patches > 0:
                self._random_patches = True
                self._patches_perslice = random_patches
                self._calculate_random_patch_indexes()
            else:
                self._calculate_patch_indexes()
                self._patches_perslice = np.cumsum([len(p) for p in self._patch_indexes])
        else:
            assert isinstance(reference_dataset, ImagePatchesLoader3D)
            self._patch_indexes = reference_dataset._patch_indexes
            self._random_patches = reference_dataset._random_patches
            self._patches_perslice = reference_dataset._patches_perslice

        # Pre-shuffle the patch indexes so that each mini-batch would have similar data-statistics as the training input
        if self._pre_shuffle:
            assert self._random_patches == 0, "Pre-shuffle cannot be used with random patches."
            self._shuffle_index_arr = np.arange(self.__len__())
            np.random.shuffle(self._shuffle_index_arr)
            self._inverse_shuffle_arr = self._shuffle_index_arr.argsort()


    def _calculate_patch_indexes(self):
        # Clear existing indexes first
        self._patch_indexes = []

        # Set up corner indexes based on selected axes
        X, Y, Z = self._axis

        for i, dat in enumerate(self._base_dataset):
            Xlen, Ylen, Zlen = dat.shape[X], dat.shape[Y], dat.shape[Z]
            Xpat, Ypat, Zpat = self._patch_size
            Xstr, Ystr, Zstr = self._patch_stride

            # Max partitions
            nX = (Xlen - Xpat) // Xstr
            nY = (Ylen - Ypat) // Ystr
            nZ = (Zlen - Zpat) // Zstr

            # Division residual
            resX = (Xlen - Xpat) % Xstr
            resY = (Ylen - Ypat) % Ystr
            resZ = (Zlen - Zpat) % Zstr

            if self._include_last_patch:
                if resX > 0:
                    nX += 1
                if resY > 0:
                    nY += 1
                if resZ > 0:
                    nZ += 1

            # Transform the grid by scaling
            x, y, z = np.meshgrid(np.arange(nX), np.arange(nY), np.arange(nZ))
            x, y, z = np.clip(x * Xpat, 0, Xlen - Xpat + 1), \
                      np.clip(y * Ypat, 0, Ylen - Ypat + 1), \
                      np.clip(z * Zpat, 0, Zlen - Zpat + 1)

            self._patch_indexes.append(zip(x.flatten(), y.flatten(), z.flatten()))

    def size(self, val=None):
        newsize = list(self._unit_dimension)
        for i in xrange(len(newsize)):
            if i == self._axis[0] % self._patch_ndim:
                newsize[i] = self._patch_size[0]
            elif i == self._axis[1] % self._patch_ndim:
                newsize[i] = self._patch_size[1]
            elif i == self._axis[2] % self._patch_ndim:
                newsize[i] = self._patch_size[2]
        size = [self.__len__()] + newsize
        if val is None:
            return size
        else:
            return size[val]

    def piece_patches(self, inpatches):
        if isinstance(inpatches, list):
            length = np.sum([len(x) for x in inpatches])
            channels = inpatches[0].size()[1]
            batch_size = inpatches[0].size()[0]
            LIST_MODE = True
        else:
            LIST_MODE = False
            length = len(inpatches)
            channels = inpatches.size()[1]
        if length != self.__len__():
            print "Warning! Size mismatch: " + str(len(inpatches)) + ',' + str(self.__len__())

        temp_slice = torch.cat([torch.zeros(self._base_dataset.data.shape, dtype=torch.float)
                                  for i in xrange(channels)], dim=1)
        temp_slice[:,0] = 1E-6 # This forces all the un processed slices to have null label.
        count = torch.zeros(temp_slice.size(), dtype=torch.int16)

        if self._random_patches:
            for j, p in enumerate(self._patch_indexes):
                i = j // self._patches_perslice
                indexes = []
                for k in xrange(count.ndimension()):
                    if k == self._axis[0] % count.ndimension():
                        indexes.append(slice(p[0], p[0] + self._patch_size[0]))
                    elif k == self._axis[1] % count.ndimension():
                        indexes.append(slice(p[1], p[1] + self._patch_size[1]))
                    elif k == self._axis[2] % count.ndimension():
                        indexes.append(slice(p[2], p[2] + self._patch_size[2]))
                    elif k == 0 and count.ndimension() == 4: # Batch dimension
                        indexes.append(slice(i, i+1))
                    else:
                        indexes.append(slice(None))

                if LIST_MODE:
                    index_1 = j % len(inpatches[0])
                    index_2 = j // len(inpatches[0])
                    temp_slice[indexes] += inpatches[index_2][index_1]
                else:
                    temp_slice[indexes] += inpatches[j]
                count[indexes] += 1
        else:
            for i in xrange(len(self._base_dataset)):
                for j, p in enumerate(self._patch_indexes):
                    indexes = []
                    for k in xrange(count.ndimension()):
                        if k == self._axis[0] % count.ndimension():
                            indexes.append(slice(p[0], p[0] + self._patch_size[0]))
                        elif k == self._axis[1] % count.ndimension():
                            indexes.append(slice(p[1], p[1] + self._patch_size[1]))
                        elif k == self._axis[2] % count.ndimension():
                            indexes.append(slice(p[2], p[2] + self._patch_size[2]))
                        elif k == 0 and count.ndimension() == 4: # Batch dimension
                            indexes.append(slice(i, i+1))
                        else:
                            indexes.append(slice(None))
                    if self._pre_shuffle:
                        inpatches_index = self._inverse_shuffle_arr[i * len(self._patch_indexes) + j]
                        if LIST_MODE:
                            index_1 = inpatches_index % len(inpatches[0])
                            index_2 = inpatches_index // len(inpatches[0])
                            temp_slice[indexes] += inpatches[index_2][index_1]
                        else:
                            temp_slice[indexes] += inpatches[inpatches_index]
                    else:
                        inpatches_index = i * len(self._patch_indexes) + j
                        if LIST_MODE:
                            index_1 = inpatches_index % len(inpatches[0])
                            index_2 = inpatches_index // len(inpatches[0])
                            temp_slice[indexes] += inpatches[index_2][index_1]
                        else:
                            temp_slice[indexes] += inpatches[inpatches_index]
                    count[indexes] += 1

        count[count == 0] = 1   # Prevent division by zero
        temp_slice /= count.float()
        return tensor(temp_slice)


    def Write(self, slices, outputdir, prefix=''):
        self._base_dataset.Write(slices, outputdir, prefix)


    def __len__(self):
        if not self._random_patches:
            return sum([len(p) for p in self._patch_indexes])
        else:
            return len(self._patch_indexes) * len(self._base_dataset)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start if not item.start is None else 0
            stop = item.stop if not item.stop is None else self.__len__()
            step = item.step if not item.step is None else 1
            return stack([self.__getitem__(i) for i in xrange(start, stop, step)], 0)
        else:
            if item >= self.__len__():
                raise IndexError("Index out of range! Requesting %s, length is %s"%(item, self.__len__()))

            # prevent overflow when meeting negative numbers
            item = item % self.__len__()

            # map item to shuffled list
            if self._pre_shuffle:
                item = self._shuffle_index_arr[item]


            slice_index = np.argmax(item < self._patches_perslice)
            patch_index = item - self._patches_perslice[slice_index - 1] if slice_index > 0 else item

            p = self._patch_indexes[slice_index][patch_index]
            s = self._base_dataset[slice_index]

            indexes = []
            for i in xrange(self._patch_ndim):
                if i == self._axis[0] % self._patch_ndim:    # in case user use negative index
                    indexes.append(slice(p[0], p[0] + self._patch_size[0]))
                elif i == self._axis[1] % self._patch_ndim:
                    indexes.append(slice(p[1], p[1] + self._patch_size[1]))
                elif i == self._axis[2] % self._patch_ndim:
                    indexes.append(slice(p[2], p[2] + self._patch_size[2]))
                else:
                    indexes.append(slice(None))

            out = s[indexes]
            while out.dim() < 4:
                out = out.unsqueeze(0)
            return out

