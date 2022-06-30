import torchio
import inspect
from .pmi_dataloader_base import PMIDataLoaderBase
from .. import med_img_dataset
from .computations import *
from .augmenter_factory import create_transform_compose
from .lambda_tio_adaptor import CallbackQueue

from typing import *
from pathlib import Path
from functools import partial
import torchio as tio
import multiprocessing as mpi
import re

__all__ = ['PMIImageDataLoader']

class PMIImageDataLoader(PMIDataLoaderBase):
    """
    This class load :class:ImageDataSet related image data. Customization of loading this class should inherit this
    class.

    This class is suitable in the following situations:
        * Image to image
        * Image to segmentation
        * Image super-resolution

    Attributes:
        regex (str, Optional):
            Filter for loading files. See :class:`ImageDataSet`
        idlist (str or list, Optional):
            Filter for loading files. See :class:`ImageDataSet`
        data_types (str, Optional):
            Define the type of data for input and ground-truth. Default to 'float-float'.
        patch_size (int or [int, int, int], Optional):
            Define the patch size to draw from each image. If `None`, use the maximum 2D or 3D size depending on whether
            attribute 'load_by_slice' >= 0. Default to None.
        load_by_slice (int, Optional):
            If value >= 0, images are loaded slice-by-slice, and this also specifies the slicing axis. Default to -1.
        sampler (str, Optional):
            ['weighted'|'uniform']. Default to 'uniform'.
        sampler_probmap_key (str, Optional)

    Args:
        *args: Please see parent class.
        **kwargs: Please see parent class.

    .. note::
        Attributes are defined in :func:`PMIImageDataLoader._read_params`, either read from a dictionary or an ini
        file. The current system reads the [LoaderParams].

    .. hint::
        Users are suppose to pass arguments to the super class for handling. If in doubt, look at the docs of parent
        class!


    See Also:
        :class:`PMIDataLoaderBase`
    """
    def __init__(self, *args, **kwargs):
        super(PMIImageDataLoader, self).__init__(*args, **kwargs)

    def _check_input(self):
        """Not implemented."""
        return True

    def _read_params(self, config_file=None):
        """
        Defines attributes. Called when object is created. Extra attributes are declared in super function,
        see the super class for more details. Params are read from `[LoaderParams]` section of the ini.

        Args:
            config_file (str or dict, Optional): See :func:`PMIDataLoaderBase._read_params`.

        See Also:
            * :class:`PMIDataLoaderBase`
            * :func:`PMIDataLoaderBase._read_params`
        """

        super(PMIImageDataLoader, self)._read_params(config_file)
        self._regex = self.get_from_config('Filters', 're_suffix', None)
        self._idlist = self.get_from_config('Filters', 'id_list', None)
        if isinstance(self._idlist, str):
            if self._idlist.endswith('.ini'):
                self._idlist = self.parse_ini_filelist(self._idlist, self._run_mode)
            elif self._idlist.endswith('.txt'):
                self._idlist = [r.rstrip() for r in open(self._idlist).readlines()]
            else:
                self._idlist = self._idlist.split(',')
            self._idlist.sort()
        self._exclude = self.get_from_config('Filters', 'id_exclude', None)
        if not self._exclude is None:
            self._exclude = self._exclude.split(',')
            for e in self._exclude:
                if e in self._idlist:
                    self._logger.info("Removing {} from the list as specified.".format(e))
                    self._idlist.remove(e)

        # Try to read sampling probability map
        self._probmap_dir = self.get_from_config('Data', 'prob_map_dir', None)
        self._mask_dir = self.get_from_config('Data', 'mask_dir', None)

        # Default attributes:
        default_attr = {
            'data_types': 'float-float',
            'idGlobber': "(^[a-zA-Z0-9]+)",
            'patch_size': None,
            'queue_kwargs': {},
            'sampler': 'uniform',
            'augmentation': '',
            'create_new_attribute': "",
            'patch_sampling_callback': "",
            'patch_sampling_callback_kwargs': {},
            'inf_samples_per_vol': None
        }
        self._load_default_attr(default_attr)
        self._logger.info(f"Read local attributes: {[self.__getattribute__(i) for i in default_attr]}")
        # Update some kwargs with more complex default settings
        default_queue_kwargs = {
            'max_length': 15,
            'samples_per_volume': 1,
            'num_workers': 16,
            'shuffle_subjects': True,
            'shuffle_patches':  True,
            'start_background': True,
            'verbose': True,
        }
        default_queue_kwargs.update(self.queue_kwargs)  # self.queue_kwargs is loaded by _load_default_attr
        if default_queue_kwargs['num_workers'] > mpi.cpu_count():
            default_queue_kwargs['num_workers'] = mpi.cpu_count()
        self.queue_kwargs = default_queue_kwargs

        if (self.sampler == 'weighted') & (self._probmap_dir is not None):
            self.sampler = tio.WeightedSampler(patch_size=self.patch_size, probability_map='probmap')
        elif self.patch_size != None:
            self.sampler = tio.UniformSampler(patch_size=self.patch_size)
        else:
            self.sampler = None
        self.queue_args = [self.queue_kwargs.pop(k)
                           for k in ['max_length', 'samples_per_volume']] \
                          + [self.sampler] # follows torchio's args arrangments

        # Build transform
        self.transform = None
        self.data_types = self.data_types.split('-')

    def _read_image(self, root_dir, **kwargs):
        """
        Private method for convenience.

        Args:
            root_dir (str): See :class:`med_img_dataset.ImageDataSet`
            **kwargs: See :class:`med_img_dataset.ImageDataSet`

        Raises:
            AttributeError: If there are no corresponding items in section `[LoaderParams]`.

        Returns:
            (ImageDataSet or ImageDataSetAugment): Loaded image data set.

        See Also:
            :class:`med_img_dataset.ImageDataSet`
        """
        if root_dir is None:
            self._logger.warning("Received None for root_dir arguement.")
            return None

        self._image_class = med_img_dataset.ImageDataSet
        img_data =  self._image_class(root_dir, verbose=self._verbose, debugmode=self._debug, filtermode='both',
                                      regex=self._regex, idlist=self._idlist, idGlobber=self.idGlobber, **kwargs)
        return img_data

    def _load_data_set_training(self,
                                exclude_augment: bool = False) -> tio.Queue:
        """
        Load ImageDataSet for input and segmentation. For more see :func:`create_transform()`.
        """
        if self._target_dir is None:
            raise IOError(f"Cannot load from {self._target_dir}")

        img_out = self._read_image(self._input_dir, dtype=self.data_types[0])
        gt_out = self._read_image(self._target_dir, dtype=self.data_types[1])
        mask_out = self._read_image(self._mask_dir, dtype='uint8')
        prob_out = self._prepare_probmap()

        self.data = self._prepare_data(gt_out, img_out, mask_out, prob_out)
        # Create transform
        transform = self._create_transform(exclude_augment=exclude_augment)

        # Create subjects & queue
        subjects = self._pack_data_into_subjects(self.data, transform)

        # Return the queue
        return self._create_queue(exclude_augment, subjects)

    def _pack_data_into_subjects(self, data: dict, transform):
        data_exclude_none = {k: v for k, v in data.items() if v is not None}
        subjects = [tio.Subject(**{k: v for k, v in zip(data_exclude_none.keys(), row)})
                    for row in zip(*data_exclude_none.values())]
        subjects = tio.SubjectsDataset(subjects=subjects, transform=transform)
        return subjects

    def _load_data_set_inference(self) -> [tio.Queue, tio.GridSampler] or [tio.SubjectsDataset, None]:
        """Same as :func:`_load_data_set_training` in this class, except the ground-truth is
        not loaded."""
        img_out = self._read_image(self._input_dir, dtype=self.data_types[0])
        prob_out = self._prepare_probmap()

        if not self._target_dir in (None, 'None'):
            try:
                gt_out = self._read_image(self._target_dir, dtype=self.data_types[1])
            except:
                self._logger.exception("Can't load from: {}".format(self._target_dir))
                self._logger.warning("Skipping ground-truth data loading.")
                gt_out = None
        else:
            gt_out = None

        self.data = self._prepare_data(gt_out, img_out, None, prob_out)
        # Creat transform
        transform = self._create_transform(exclude_augment = True)

        # Create subjects & queue
        subjects = self._pack_data_into_subjects(self.data, transform)

        # override the number of patches drawn in this special case
        if self.inf_samples_per_vol is not None:
            self._logger.info(f"Override `samples_per_vol` {self.queue_args[1]} with "
                              f"`inf_samples_per_vol` {self.inf_samples_per_vol}")
            self.queue_args[1] = int(self.inf_samples_per_vol)

        # No transform for subjects
        return self._create_queue(True, subjects, return_sampler=False)

    def _prepare_data(self, gt_out, img_out, mask_out, prob_out):
        """
        Convinient method to create data that will be loaded as subjects

        """
        data = {'input': img_out,
                'gt': gt_out,
                'mask': mask_out,
                'probmap': prob_out,
                'uid': img_out.get_unique_IDs()
                }
        data['orientation'] = [i.orientation for i in img_out]
        for k in ['input', 'gt', 'mask', 'probmap']:
            if isinstance(data[k], med_img_dataset.ImageDataSet):
                data[f'{k}-shape'] = data[k].get_raw_data_shape()
        return data

    def _prepare_probmap(self):
        r"""Load probability map if its specified."""
        # Load probability map if specified
        if self._probmap_dir is not None:
            self._logger.info(f"Loading probmap from: {self._probmap_dir}")
            try:
                prob_out = self._read_image(self._probmap_dir, dtype='uint32') # torchio requires Integer probmap
            except:
                self._logger.warning(f"Couldn't open probmap from: {self._probmap_dir}", no_repeat=True)
                prob_out = None
        else:
            prob_out = None
        return prob_out

    def _determine_patch_size(self, im_ref):
        r"""If `load_by_slice` of :class:`ImageDataSet` is setted, this method will calculate the patch-size
        automatically for the queue sampler."""
        if self.patch_size is not None:
            return
        else:
            ref_im_shape = list(im_ref.shape)
            if self.load_by_slice >= 0:
                ref_im_shape[self.load_by_slice] = 1
                self.patch_size = ref_im_shape
            else:
                self.patch_size = ref_im_shape

    def _create_queue(self,
                      exclude_augment: bool,
                      subjects: tio.SubjectsDataset,
                      return_sampler: Optional[bool] =False) -> [tio.Queue, tio.GridSampler] or \
                                                                [tio.SubjectsDataset, None]:
        r"""This method build the queue from the input subjects. If the queue involves a :class:`tio.GridSampler`,
        it is generally needed by the inferencer to reconstruct it back into its original dimension. Thus, an
        optional to also return the sampler is given.
        """
        # Return the queue
        if not self.patch_size is None:
            overlap = [ps // 2 for ps in self.patch_size]
            # If no probmap, return GridSampler, otherwise, return weighted sampler
            if self.data['probmap'] is None:
                sampler = tio.GridSampler(patch_size=self.patch_size, patch_overlap=overlap)
            else:
                sampler = self.sampler
        else:
            # Set queue_args and queue_kwargs to load the whole image for each object to allow for caching
            shape_of_input = subjects[0].shape

            # Reset sampler
            self.sampler = tio.UniformSampler(patch_size=shape_of_input[1:])  # first dim is batch
            self.queue_args[-1] = self.sampler

        # if exclude augment, don't shuffle
        if exclude_augment:
            queue_dict = self.queue_kwargs
            queue_dict['shuffle_subjects'] = False
        else:
            queue_dict = self.queue_kwargs

        # Create queue
        # If option to use post-sampling processing was provided, use CallbackQueue instead
        if  self.patch_sampling_callback != "":
            # check if there's illegal characters in the patch_sampling_callback
            if re.search("[\W]+", self.patch_sampling_callback.translate(str.maketrans('', '', "[], "))) is not None:
                raise AttributeError(f"You patch_sampling_callback specified ({self.patch_sampling_callback}) "
                                     f"contains illegal characters!")
            _callback_func = eval(self.patch_sampling_callback)
            _callback_func = partial(_callback_func, **self.patch_sampling_callback_kwargs)
            queue = CallbackQueue(subjects, *self.queue_args,
                                  patch_sampling_callback=_callback_func,
                                  create_new_attribute=self.create_new_attribute,
                                  **queue_dict)
        else: # Else use the normal queue
            # queue_dict.pop('patch_sampling_callback')
            # queue_dict.pop('create_new_attribute')
            queue = tio.Queue(subjects, *self.queue_args, **queue_dict)
        self._logger.debug(f"Created queue: {queue}")
        self.queue = queue

        if return_sampler:
            return queue, sampler
        else:
            return queue

    def create_aggregation_queue(self, subject: torchio.SubjectsDataset, *args, **kwargs):
        r"""Note that this function should only be invoked during inference. Typically, you don't need the
        aggregator anywhere else."""
        required_att = ('sampler', 'data', 'queue')
        for att in required_att:
            if not hasattr(self, att):
                msg += f"Attribute {att} missing. Have you run load_data() already?"
                raise AttributeError(msg)

        if isinstance(self.sampler, tio.GridSampler):
            self.sampler.set_subject(subject)
        elif isinstance(self.sampler, tio.WeightedSampler):
            _spv = self.inf_samples_per_vol if self.inf_samples_per_vol is not None \
                else self.queue.samples_per_volume
            self._logger.info(f"Setting the number of patches to sample to: "
                              f"{_spv}")
            self.sampler.set_subject(subject, _spv)
            self.queue.samples_per_volume = _spv
        else:
            msg = f"Currrently only support GridSampler and WeightedSampler, but got {type(self.sampler)}"
            raise TypeError(msg)

        # Replace subjects in queue and reset the queue
        self.queue._subjects_iterable = None
        self.queue.sampler = self.sampler
        aggregator = tio.GridAggregator(self.sampler, 'average')
        return self.queue, aggregator

    def get_sampler(self):
        return self.sampler