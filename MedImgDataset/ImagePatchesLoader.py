import torch
import torch.multiprocessing as mpi
from functools import partial
from torch import cat, stack, tensor, zeros
from torch.utils.data import Dataset
from .ImageData import ImageDataSet
from .ImageDataMultiChannel import ImageDataSetMultiChannel
from .ImageDataAugment import ImageDataSetAugment
from tqdm import *
import numpy as np
import gc

def _mpi_wrapper(roi, pos, p_indexes=None, semaphore=None, **kwargs):
    """This is required for mpi choose patches"""
    try:
        if p_indexes is None:
            print("_mpi_wrapper encounters error input configurations!")
            semaphore.release()
            return 1
        out = ImagePatchesLoader._choose_from_probability_map(roi, **kwargs)
        out = np.array(out)
        p_indexes[pos * out.shape[0]:pos * out.shape[0] + out.shape[0]].copy_(torch.from_numpy(out))
        semaphore.release()
        del out, roi
        return 0
    except Exception as e:
        semaphore.release()
        return 1

# def _mpi_wrapper(pos, dat=None, indexes=None, semaphore=Semaphore(1), **kwargs):
#     """This is required for mpi choose patches"""
#     if dat is None or indexes is None or semaphore is None:
#         # return immediately
#         print("_mpi_wrapper encounters wrong configurations.")
#         return
#
#     semaphore.accquire()
#     roi = np.copy(dat[pos][indexes].data.numpy().astype('float16'))
#
#     # normalize
#     roi = roi - roi.min()
#     roi = roi / roi.max()
#
#     # if result is 3D collapse it into 2D by averaging
#     while roi.ndim > 2:
#         roi = np.mean(roi, axis=np.argmin(roi.shape))
#     out = ImagePatchesLoader._choose_from_probability_map(roi, **kwargs)
#     del roi
#     gc.collect()
#     semaphore.release()
#     return out, pos



class ImagePatchesLoader(Dataset):
    """
    This class depends on but does not inherit the class :class:`ImageDataSet`. This class sample patches from the
    input `ImageDataSet` or its child classes if they are loaded by slices (i.e. `loadBySlice != 0`).

    There are two main loading mode for this class: 1) load by partitioning the slices evenly 2) load by randomly
    drawing equal size patches. The loaded patches are eventually stacked back into a column of patches. The channels
    of the images are inherited in each of the patches.

    For each loading modes, there are options that decide the loading sequence.

    Attributes:
        has_reference (bool):
            (**Random mode**) Not set explicited. Indicate whether this object was referenced by another object.
        unit_dimension (int):
            Number of dimensions the input data has.
        slice_dim (int or list of int):
            Number of dimensions each input element has. In generally equals to `unit_dimension - 1` but there are
            exceptions when one of the input dimension is actually a vitual axis with length be 1.

    Args:
        patch_size (int or list of int):
            Size of extracted patch. If its an `int`, assume patches are square patches.
        pre_shuffle (bool, Optional):
            Specify whether to shuffle patches order before stacking them up. Is usually helpful to maintain a
            consistent value statistics such that the batch-norms trained with random order functions as effectively
            during inference. This should be set to 'True' during inference but 'False' during training in general.
            Default to 'False'.
        axis ([int, int], Optional):
            Patches are drawn from 2D slices parallel to the specified option. Default to the last two axes `[-2,-1]`.
        patch_stride (int or list of int, Optional):
            (**Uniform mode**) Decide the stride between adjacent patches, if value smaller then patch dimension,
            the output patches will overlap. For negative value, patches stride will be set to be identical to patch
            dimension. Ignored if `random_patches > 0`. Default to `-1`.
        include_last_patch (bool, Optional):
            (**Uniform mode**) Whether to include last patches when stride and patch dimensions does not divide input
            image completely such that all voxels in an image will be present in the patches stack. Inored if
            `random_patches > 0`. Default to `True`.
        random_patches (int, Optional):
            (**Random mode**) Number of patches to sample from the image per slices. Ignored if `<= 0`. Default to `-1`.
        random_from_distribution (callalble, Optional):
            (**Random mode**) A callable function can be submitted to calculate the sampling probability of each
            slices with the slice as the function argument. The function should return an array identical to slice
            dimension that summed to 1. Otherwise, each voxel has the same probability to be sampled. Ignored if is
            `None`. Default to `None`.
        renew_index (bool, Optional):
            (**Random mode**) Specify whether to sample again after each Epoch. Defaul to `False`.
        reference_dataset (ImagePatchesLoader, Optional):
            (**Random mode**) This options allows another ImagePathcesLoader object to reference this object when
            computing sampling index. If not `None`, ts sampling index of the referencee will be identical to this
            object and will renew together in cases `renew_index == True`. Useful when performing segmentation as
            you want the ground-truth to has the same patches sampling location. Default to 'None'.

        base_data (ImageDataSet): Input to be sampled.

    Examples:

        >>> from MedImgDataset import ImageDataSet, ImageDataSetAugment, ImagePatchesLoader
        >>>
        >>> # Read images
        >>> imset = ImageDataSet('.', verbose=True)
        >>> imset_aug = ImageDataSetAugment('.', verbose=True, aug_factor=5, augmentator=)
        >>>
        >>> # This dataset has same augmentation state as imset_aug
        >>> segset_aug = ImageDataSetAugment('./Seg', verbose=True, aug_factor=5,
        >>>                                  reference_dataset=imset_aug, is_seg=True)
        >>>
        >>> # Standard patches
        >>> img_patches = ImagePatchesLoader(imset, patch_size=64, random_patches=20)
        >>>
        >>> # Augment first, then draw patches
        >>> img_aug_patches = ImagePatchesLoader(imset_aug, patch_size=64, random_patches=20)
        >>> # This patches sample the same patches at locations same as img_aug_patches
        >>> segseg_aug_patches = ImagePatchesLoader(segseg_aug, patch_size=64, random_patches=20,
        >>>                                         reference_dataset=img_aug_patches)




    .. note::
        This class has the ability to align random states of multiple objects using the argument `reference_dataset`.
        Users would like to do this in times that the patches are extracted for segmentation. Do note that python,
        unlike c++, passes arguements by referneces, so every time you update one of the chained object, the random
        state is passed on to others top down. For instance, if you chained `b->a` and `c->a`, `b` and `c` will
        change state if you changes `a` (e.g. through calling batch_done_callback), but `c` will remains if you
        changes `b`. Currently, this is mitigated through using np.ndarray as a psuedo pointer.

    .. hint::
        You want to set `pre_shuffle` to `True` during inference and `False` during training because by default,
        the training solver shuffles the data before sampling but the inferencer won't to keep track of things.




    """
    def __init__(self, base_dataset, patch_size, patch_stride=-1, include_last_patch=True,
                 axis=None, reference_dataset=None, pre_shuffle=False, random_patches=-1,
                 random_from_distribution=None, renew_index=True):
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
        xy = list(zip(xy[0].flatten(), xy[1].flatten()))

        if np.isclose(prob.sum(), 0.):
            prob = np.ones_like(prob)
        prob = prob / prob.sum()

        try:
            choices = np.random.choice(len(xy), p=prob.flatten(), size=pps)
        except Exception as e:
            print(roi.shape, len(xy), prob.flatten().shape, e)
        out = [xy[c] for c in choices]
        del xy, choices, prob, roi
        return out

    def _calculate_random_patch_indexes(self):
        """Calculated the corner index of the patches."""
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
        except Exception as e:
            self._patch_indexes = patch_indexes
        pass


    def _mpi_patch_index_callback(self, pos, indexes):
        try:
            self._patch_indexes[pos:pos + len(indexes)] = np.array(indexes)
        except Exception as e:
            print("Error! Callback not working correctly! {}".format(e))

    def _sample_patches_from_distribution(self):
        """Sample patches with probability distribution."""
        # Use multiprocessing
        # import torch.multiprocessing as mpi
        # from functools import partial

        # Set up arguments
        X, Y = self._axis
        corner_range = [self._unit_dimension[X] - self._patch_size[0],
                        self._unit_dimension[Y] - self._patch_size[1]]
        func = self._random_from_distrib

        indexes = []
        for j in range(self._slice_dim):
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


        # release memory
        if not self._patch_indexes is None:
            del self._patch_indexes

        self._patch_indexes = np.zeros([len(self._base_dataset) * self._patch_perslice, 2], dtype='int16')
        self._patch_indexes = torch.from_numpy(self._patch_indexes).share_memory_()

        # Non mpi
        # wrapped_callable = partial(_mpi_wrapper, func=func, pps=self._patch_perslice, indexes=indexes,
        #                            dat=self._base_dataset)
        # for i in tqdm(range(len(self._base_dataset)), desc="Sampling job.", total=len(self._base_dataset)):
        #     p_indexes, pos = wrapped_callable(i)
        #     p_indexes = np.array(p_indexes)
        #     self._patch_indexes[pos * p_indexes.shape[0]:pos * p_indexes.shape[0] + p_indexes.shape[0]] = p_indexes

        # Semaphore will lock the memory consumption to a standard
        sema = mpi.Manager().Semaphore(15)
        pool = mpi.Pool(mpi.cpu_count())
        ps = []
        for i in tqdm(range(len(self._base_dataset)), total=len(self._base_dataset)):
            dat = self._base_dataset[i]
            sema.acquire(timeout=60)
            roi = np.copy(dat[indexes].data.squeeze().numpy())
            roi = (roi - roi.min()) / (roi.max() - roi.min())
            roi = roi.astype('float16')

            p = pool.apply_async(partial(_mpi_wrapper,
                                         p_indexes = self._patch_indexes,
                                         func=func,
                                         semaphore=sema,
                                         pps=self._patch_perslice),
                                 (roi, i)
                                 )
            ps.append(p)
            del roi

        for p in ps:
            p.get(20) # Prevent some process died without a trace and lock the thread

        pool.close()
        pool.join()

        # speacial treatment required if baseclass has augmentator
        if isinstance(self._base_dataset, ImageDataSetAugment):
            self._base_dataset._update_each_epoch = temp_flag
            self._base_dataset._call_count = 0
        del sema


    def _calculate_patch_indexes(self):
        """Calcualted patches index for uniform patches."""
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
        nX = (Xlen - Xpat) // Xstr
        nY = (Ylen - Ypat) // Ystr

        # Division residual
        resX = (Xlen - Xpat) % Xstr
        resY = (Ylen - Ypat) % Ystr

        func = lambda x, y: (x * Xstr, y * Ystr)

        for i in range(nX + 1):
            for j in range(nY + 1):
                self._patch_indexes.append(func(i, j))

                if resY != 0 and j == nY and self._include_last_patch:
                    self._patch_indexes.append([i * Xstr, j * Ystr + resY])
            if resX != 0 and i == nX and self._include_last_patch:
                for k in range(nY + 1):
                    self._patch_indexes.append([i * Xstr + resX, k * Ystr])
            if resX != 0 and resY != 0 and i == nX and j == nY and self._include_last_patch:
                self._patch_indexes.append([i * Xstr + resX, j * Ystr + resY])
        self._patch_indexes = np.array(self._patch_indexes)

    def size(self, val=None):
        """Requried by pytorch."""
        newsize = list(self._unit_dimension)
        for i in range(len(newsize)):
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
        """
        Pieces patches back into one accordining to where it was sampled. Usefult for cases like image de-noising or
        segmentation.

        Args:
            inpatches (torch.tensor):
                A tensor that should have a dimension identical to `self.data`.

        Returns:
            (torch.tensor):
                A tensor that has a shape identical to `self.basedata.data`

        """
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
            print("Warning! Size mismatch: " + str(len(inpatches)) + ',' + str(self.__len__()))

        temp_slice = torch.cat([torch.zeros(self._base_dataset.data.shape, dtype=torch.float)
                                  for i in range(channels)], dim=1)
        temp_slice[:,0] = 1E-6 # This forces all the un processed slices to have null label.
        count = torch.zeros(temp_slice.size(), dtype=torch.int16)

        if self._random_patches:
            for j, p in enumerate(self._patch_indexes):
                i = j // self._patch_perslice
                indexes = []
                for k in range(count.ndimension()):
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
            for i in range(len(self._base_dataset)):
                for j, p in enumerate(self._patch_indexes):
                    indexes = []
                    for k in range(count.ndimension()):
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
        """Called after `__getitem__` is called for `__len__` number of time."""
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
            return stack([self.__getitem__(i) for i in range(start, stop, step)], 0)
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
                slice_index = item // len(self._patch_indexes)
                patch_index = item % len(self._patch_indexes)

            p = self._patch_indexes[patch_index]

            indexes = []
            for i in range(self._slice_dim):
                if i == self._axis[0] % self._slice_dim:    # in case user use negative index
                    indexes.append(slice(p[0], p[0] + self._patch_size[0]))
                elif i == self._axis[1] % self._slice_dim:
                    indexes.append(slice(p[1], p[1] + self._patch_size[1]))
                else:
                    indexes.append(slice(None))
            return self._base_dataset[slice_index][indexes]

