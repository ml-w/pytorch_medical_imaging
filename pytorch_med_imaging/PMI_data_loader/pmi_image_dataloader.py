from .pmi_dataloader_base import PMIDataLoaderBase
from .. import med_img_dataset
from .computations import *
from .augmenter_factory import create_transform_compose
from .lambda_tio_adaptor import CallbackQueue

from pathlib import Path
from functools import partial
import torchio as tio
import multiprocessing as mpi

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
            'patch_sampling_callback_kwargs': {}
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
                                exclude_augment: bool = False):
        """
        Load ImageDataSet for input and segmentation.
        """
        if self._target_dir is None:
            raise IOError(f"Cannot load from {self._target_dir}")

        img_out = self._read_image(self._input_dir, dtype=self.data_types[0])
        gt_out = self._read_image(self._target_dir, dtype=self.data_types[1])
        mask_out = self._read_image(self._mask_dir, dtype='uint8')
        prob_out = self._prepare_probmap()

        self.data = self._prepare_data(gt_out, img_out, mask_out, prob_out)
        # Create transform
        self._create_transform(exclude_augment=exclude_augment)

        # Create subjects & queue
        data_exclude_none = {k: v for k, v in self.data.items() if v is not None}
        subjects = [tio.Subject(**{k:v for k, v in zip(data_exclude_none.keys(), row)})
                    for row in zip(*data_exclude_none.values())]
        subjects = tio.SubjectsDataset(subjects=subjects, transform=self.transform)

        # Return the queue
        return self._create_queue(exclude_augment, subjects)


    def _load_data_set_inference(self) -> [tio.Queue, tio.GridSampler] or [tio.SubjectsDataset, None]:
        """Same as :func:`_load_data_set_training` in this class."""
        img_out = self._read_image(self._input_dir, dtype=self.data_types[0])
        prob_out = self._prepare_probmap()

        if not self._target_dir is None:
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
        self._create_transform(exclude_augment = True)

        # Create subjects & queue
        data_exclude_none = {k: v for k, v in self.data.items() if v is not None}
        subjects = [tio.Subject(**{k:v for k, v in zip(data_exclude_none.keys(), row)})
                    for row in zip(*data_exclude_none.values())]
        subjects = tio.SubjectsDataset(subjects=subjects, transform=self.transform)

        # No transform for subjects
        r = [subjects]
        r.extend(list(self._create_queue(True, subjects, return_sampler=True)))
        return r

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
        if self.patch_size is not None:
            return
        else:
            ref_im_shape = list(im_ref.shape)
            if self.load_by_slice >= 0:
                ref_im_shape[self.load_by_slice] = 1
                self.patch_size = ref_im_shape
            else:
                self.patch_size = ref_im_shape

    def _create_queue(self, exclude_augment, subjects, return_sampler=False):
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

        if return_sampler:
            return queue, sampler
        else:
            return queue

