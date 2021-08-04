from .PMIDataLoaderBase import PMIDataLoaderBase
from .. import med_img_dataset

import re
import torchio as tio

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

        # Default attributes:
        default_attr = {
            'data_types': 'float-float',
            'idGlobber': "(^[a-zA-Z0-9]+)",
            'patch_size': None,
            'queue_kwargs': {},
            'sampler': 'uniform',
            'augmentation': False
        }
        self._load_default_attr(default_attr)
        # Update some kwargs with more complex default settings
        default_queue_kwargs = {
                    'max_length': 150,
                    'samples_per_volume': 25,
                    'num_workers': 16,
                    'shuffle_subjects': True,
                    'shuffle_patches':  True,
                    'start_background': True,
                    'verbose': True
        }
        default_queue_kwargs.update(self.queue_kwargs)
        self.queue_kwargs = default_queue_kwargs
        if (self.sampler == 'weighted') & (self._probmap_dir is not None):
            self.sampler = tio.WeightedSampler(patch_size=self.patch_size, probability_map='probmap')
        else:
            self.sampler = tio.UniformSampler(patch_size=self.patch_size)
        self.queue_args = [self.queue_kwargs.pop(k)
                           for k in ['max_length', 'samples_per_volume']] \
                          + [self.sampler] # follows torchio's args arrangments
        # Build transform
        self.transform = tio.Compose((
            tio.ToCanonical(),
            tio.CropOrPad(),
            tio.RandomAffine(scales=[0.9, 1.1], degress=10),
            tio.RandomFlip('lr'),
            tio.RandomNoise(std=(0, 8))
        )) if self.augmentation else None

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
        self._image_class = med_img_dataset.ImageDataSet
        img_data =  self._image_class(root_dir, verbose=self._verbose, debugmode=self._debug, filtermode='both',
                                      regex=self._regex, idlist=self._idlist, idGlobber=self.idGlobber, **kwargs)
        return img_data

    def _load_data_set_training(self):
        """
        Load ImageDataSet for input and segmentation.
        """
        if self._target_dir is None:
            raise AttributeError("Object failed to load _target_dir.")

        img_out = self._read_image(self._input_dir, dtype=self.data_types[0])
        gt_out = self._read_image(self._target_dir, dtype=self.data_types[1])
        prob_out = self._prepare_probmap()

        self.data = {'input':   img_out,
                     'gt':      gt_out,
                     'probmap': prob_out
                    }

        # Create subjects & queue
        if prob_out is not None:
            subjects = [tio.Subject(input=a, gt=b, probmap=c)
                        for a, b, c in zip(img_out.data, gt_out.data, prob_out.data)]
        else:
            subjects = [tio.Subject(input=a, gt=b)
                        for a, b in zip(img_out.data, gt_out.data)]
        subjects = tio.SubjectsDataset(subjects=subjects, transform=self.transform)

        # Return the queue
        if not self.patch_size is None:
            queue = tio.Queue(subjects, *self.queue_args, **self.queue_kwargs)
            return queue
        else:
            return subjects


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

        self.data = {'input': img_out,
                     'gt': gt_out,
                     'probmap': prob_out}

        if prob_out is not None:
            subjects = [tio.Subject(input=a, probmap=b) for a, b in zip(img_out.data, prob_out.data)]
        else:
            subjects = [tio.Subject(input=a) for a in img_out.data]

        # No transform for subjects
        subjects = tio.SubjectsDataset(subjects=subjects)
        if not self.patch_size is None:
            overlap = [ps // 2 for ps in self.patch_size]
            # If no probmap, return GridSampler, otherwise, return weighted sampler
            if self.data['probmap'] is None:
                sampler = tio.GridSampler(patch_size=self.patch_size, patch_overlap=overlap)
            else:
                sampler = self.sampler
            return subjects, sampler
        else:
            return subjects, None

    def _prepare_probmap(self):
        r"""Load probability map if its specified."""
         # Load probability map if specified
        if self._probmap_dir is not None:
            self._logger.info("Loading probmap.")
            prob_out = self._read_image(self._probmap_dir, dtype=float)
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

