import torch
from torch import cat, stack, tensor, zeros
from torch.utils.data import Dataset
from ImageData import ImageDataSet
from ImageDataMultiChannel import ImageDataSetMultiChannel
from ImageDataAugment import ImageDataSetAugment
from tqdm import *
import numpy as np


def _mpi_wrapper(arg, **kwargs):
    """This is required for mpi choose patches"""
    return ImagePatchesLoader._choose_from_probability_map(arg, **kwargs)

class ImagePatchesLoader(Dataset):
    def __init__(self, base_dataset, patch_size, patch_stride=-1, include_last_patch=True,
                 axis=None, reference_dataset=None, pre_shuffle=False, random_patches=-1,
                 random_from_distribution=None, renew_index=True):
        """ImagePatchesLoader(self, base_dataset, patch_size, patch_stride, include_last_patch=True,
                 axis=None, reference_dataset=None, pre_shuffle=False, random_patches=-1,
                 random_from_distribution=None) --> ImagePatchesLoader
        """
        super(ImagePatchesLoader, self).__init__()

        assert axis is None or len(axis) == 2, \
            "Axis argument should contain the two axises that forms the base image."
        assert isinstance(base_dataset, ImageDataSet) or isinstance(base_dataset, ImageDataSetMultiChannel)
        assert reference_dataset is None or isinstance(reference_dataset, ImagePatchesLoader)
        assert not (patch_stride == -1 and random_patches == -1), \
            "You must select a patch stride if not using random patches."

        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]
        if isinstance(patch_stride, int):
            patch_stride = [patch_stride, patch_stride]

        self._pre_shuffle = pre_shuffle     # If pre-shuffle is True, patch indexes will be shuffled.
        self._base_dataset = base_dataset   # Base ImageDataSet.
        self._patch_size = patch_size       # Dimension of each patches.
        self._patch_stride = patch_stride   # Stide of patch extraction windows, ignored if random_patches is True.
        self._axis = axis if not axis is None else [-2, -1] # Default to the last two axes.
        self._include_last_patch= include_last_patch
        self._patch_indexes = []            # [(xmin, ymin), ... ], corners of the patches.
        self._random_patches = False
        self._random_counter = 0            # This counter is used to re-calculate patch corner after each epoch
        self._random_from_distrib = \
            random_from_distribution        # This is a function to covert a region of an image to a
                                            # probability map
        self._renew_index = renew_index     # If this is true, index will be renewed after each epoch
        self._has_reference = False
        self.data = self._base_dataset.data

        # check axis
        assert len(self._patch_size) <= 2, "Support 2D patches only."
        assert len(self._patch_stride) <= 2, "Support 2D patches only."
        assert self._base_dataset[0].dim() <= len(self._axis) + 1, "Wrong dimension settings."
        self._unit_dimension = self._base_dataset[0].size()
        self._slice_dim = self._base_dataset[0].dim()

        if reference_dataset is None:
            if random_patches > 0:
                self._random_patches = True
                self._patch_perslice = random_patches
                if callable(random_from_distribution):
                    # input argument `random_from_distribution` should return 2D numpy array denoting
                    # probability map of index selection.
                    self._sample_patches_from_distribution()
                else:
                    self._calculate_random_patch_indexes()
            else:
                self._calculate_patch_indexes()
            pass
        else:
            assert isinstance(reference_dataset, ImagePatchesLoader)
            self._patch_indexes = reference_dataset._patch_indexes
            self._random_patches = reference_dataset._random_patches
            self._patch_perslice = reference_dataset._patch_perslice
            self._has_reference = True

        # Pre-shuffle the patch indexes so that each mini-batch would have similar data-statistics as the training input
        if self._pre_shuffle:
            assert self._random_patches == 0, "Pre-shuffle cannot be used with random patches."
            self._shuffle_index_arr = np.arange(self.__len__())
            np.random.shuffle(self._shuffle_index_arr)
            self._inverse_shuffle_arr = self._shuffle_index_arr.argsort()

    @staticmethod
    def _choose_from_probability_map(roi, func=None, pps=None):
        """This function written for parallel computation."""
        # Calculate probability by input function
        prob = func(roi)

        # Sample accordingly
        xy = np.meshgrid(np.arange(prob.shape[0]), np.arange(prob.shape[1]))
        xy = zip(xy[0].flatten(), xy[1].flatten())

        if np.isclose(prob.sum(), 0.):
            prob = np.ones_like(prob)
        prob = prob / prob.sum()

        try:
            choices = np.random.choice(len(xy), p=prob.flatten(), size=pps)
        except Exception as e:
            print roi, len(xy), prob.flatten(), e.message
        return [xy[c] for c in choices]

    def _calculate_random_patch_indexes(self):
        # Don't touch anything if this is referencing something
        if self._has_reference:
            return

        X, Y = self._axis
        corner_range = [self._unit_dimension[X] - self._patch_size[0],
                        self._unit_dimension[Y] - self._patch_size[1]]

        patch_indexes = np.stack([np.random.randint(0, corner_range[0], size=[self._patch_perslice * len(self._base_dataset)]),
                                  np.random.randint(0, corner_range[1], size=[self._patch_perslice * len(self._base_dataset)])],
                                 -1)

        try:
            np.copyto(self._patch_indexes, patch_indexes)
        except Exception, e:
            self._patch_indexes = patch_indexes
        pass

    def _sample_patches_from_distribution(self):
        # Use multiprocessing
        import multiprocessing as mpi
        from functools import partial

        # Set up arguments
        X, Y = self._axis
        corner_range = [self._unit_dimension[X] - self._patch_size[0],
                        self._unit_dimension[Y] - self._patch_size[1]]
        func = self._random_from_distrib

        indexes = []
        for j in xrange(self._slice_dim):
            if j == self._axis[0] % self._slice_dim:    # in case user use negative index
                indexes.append(slice(self._patch_size[0] // 2,
                                     corner_range[0] + self._patch_size[0] - self._patch_size[0] // 2))
            elif j == self._axis[1] % self._slice_dim:
                indexes.append(slice(self._patch_size[1] // 2,
                                     corner_range[1] + self._patch_size[1] - self._patch_size[1] // 2))
            else:
                indexes.append(slice(None))

        # speacial treatment required if baseclass has augmentator
        if isinstance(self._base_dataset, ImageDataSetAugment):
            temp_flag = self._base_dataset._update_each_epoch
            self._base_dataset._update_each_epoch = False


        pool = mpi.Pool(mpi.cpu_count())
        rois = []
        patch_indexes = []
        for i, dat in tqdm(enumerate(self._base_dataset), desc="Sampling jobs", total=len(self._base_dataset)):
            # extract target region
            ori_roi = torch.clone(dat[indexes])

            # convert to numpy
            if isinstance(ori_roi, torch.Tensor):
                roi = ori_roi.data.squeeze().numpy()
            elif not isinstance(ori_roi, np.ndarray):
                roi = np.array(ori_roi)
            else:
                roi = np.copy(ori_roi)

            # if result is 3D collapse it into 2D by averaging
            while roi.ndim > 2:
                roi = np.mean(roi, axis=np.argmin(roi.shape))

            rois.append(roi)

        # Non-MPI
        # for roi in tqdm(rois, desc="ROI sampling"):
        #     patch_indexes.extend(_mpi_wrapper(roi, func=func, pps=self._patch_perslice))

        # # MPI
        results = pool.map_async(partial(_mpi_wrapper,
                                          func=func,
                                         pps=self._patch_perslice),
                                 rois)
        pool.close()
        pool.join()

        for r in results.get():
            patch_indexes.extend(r)

        # speacial treatment required if baseclass has augmentator
        if isinstance(self._base_dataset, ImageDataSetAugment):
            self._base_dataset._update_each_epoch = temp_flag
            self._base_dataset._call_count = 0

        self._patch_indexes = np.array(patch_indexes)
        del rois



    def _calculate_patch_indexes(self):
        # Don't touch anything if this is referencing something
        if self._has_reference:
            return

        # Clear existing indexes first
        self._patch_indexes = []

        # Set up corner indexes based on selected axes
        X, Y = self._axis

        Xlen, Ylen = self._unit_dimension[X], self._unit_dimension[Y]
        Xpat, Ypat = self._patch_size
        Xstr, Ystr = self._patch_stride

        # Max partitions
        nX = (Xlen - Xpat) / Xstr
        nY = (Ylen - Ypat) / Ystr

        # Division residual
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
        self._patch_indexes = np.array(self._patch_indexes)

    def size(self, val=None):
        newsize = list(self._unit_dimension)
        for i in xrange(len(newsize)):
            if i == self._axis[0] % self._slice_dim:
                newsize[i] = self._patch_size[0]
            elif i == self._axis[1] % self._slice_dim:
                newsize[i] = self._patch_size[1]
        size = [self.__len__()] + newsize
        if val is None:
            return size
        else:
            return size[val]

    def piece_patches(self, inpatches):
        if isinstance(inpatches, list):
            length = np.sum([len(x) for x in inpatches])
            channels = inpatches[0].size()[1]
            batch_size = len(inpatches[0])
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
                i = j // self._patch_perslice
                indexes = []
                for k in xrange(count.ndimension()):
                    if k == self._axis[0] % count.ndimension():
                        indexes.append(slice(p[0], p[0] + self._patch_size[0]))
                    elif k == self._axis[1] % count.ndimension():
                        indexes.append(slice(p[1], p[1] + self._patch_size[1]))
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


    def get_unique_values(self):
        assert isinstance(self._base_dataset, ImageDataSet), "This class must be based on ImageDataSet to use this method."
        return self._base_dataset.get_unique_values()

    def get_unique_values_n_counts(self):
        assert isinstance(self._base_dataset, ImageDataSet), "This class must be based on ImageDataSet to use this method."
        return self._base_dataset.get_unique_values_n_counts()

    def batch_done_callback(self):
        if self._renew_index:
            if callable(self._random_from_distrib):
                self._sample_patches_from_distribution()
            else:
                self._calculate_random_patch_indexes()
            self._random_counter = 0

        try:
            self._base_dataset.batch_done_callback()
        except:
            pass

    def __len__(self):
        return len(self._patch_indexes) * len(self._base_dataset) if not self._random_patches \
            else len(self._patch_indexes)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start = item.start if not item.start is None else 0
            stop = item.stop if not item.stop is None else self.__len__()
            step = item.step if not item.step is None else 1
            return stack([self.__getitem__(i) for i in xrange(start, stop, step)], 0)
        else:
            # map item to shuffled list
            if self._pre_shuffle:
                item = self._shuffle_index_arr[item]

            if self._random_patches:
                self._random_counter += 1
                slice_index = item // self._patch_perslice
                patch_index = item

                # simple trick to update patch indexes after each epoch
                if self._random_counter >= self.__len__() and self._renew_index:
                    self.batch_done_callback()
            else:
                slice_index = item / len(self._patch_indexes)
                patch_index = item % len(self._patch_indexes)

            p = self._patch_indexes[patch_index]
            s = self._base_dataset[slice_index]

            indexes = []
            for i in xrange(self._slice_dim):
                if i == self._axis[0] % self._slice_dim:    # in case user use negative index
                    indexes.append(slice(p[0], p[0] + self._patch_size[0]))
                elif i == self._axis[1] % self._slice_dim:
                    indexes.append(slice(p[1], p[1] + self._patch_size[1]))
                else:
                    indexes.append(slice(None))
            return s[indexes]

