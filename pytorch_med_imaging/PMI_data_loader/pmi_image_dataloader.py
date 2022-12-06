import torchio
from .pmi_dataloader_base import PMIDataLoaderBase, PMIDataLoaderBaseCFG
from .. import med_img_dataset
from .lambda_tio_adaptor import CallbackQueue
from typing import *
from functools import partial
import torchio as tio
import multiprocessing as mpi
import re

__all__ = ['PMIImageDataLoader', 'PMIImageDataLoaderCFG']

class PMIImageDataLoaderCFG(PMIDataLoaderBaseCFG):
    r"""Configuration for ``PMIImageDataLoader

    Attributes:
        data_types (iterable):
            Data type of input and ground-truth
        sampler (str):
            Determine the ``tio.Sampler`` used to sample the images. Support ['weighted'|'uniform'|'grid'] currently.
        sampler_kwargs (dict):
            The kwargs passed to ``tio.Sampler``. For 'weighted', key ``patch_size`` and ``prob_map`` is required. For
            'uniform', only ``patch_size`` is required.
        augmentation (str):
            Path to yaml file to create the ``tio.Compose`` transform.
        patch_sampling_callback (Callable, str):
            A function that is called after patch sampling to generate new data from sampled patches. For example, if
            texture features is required after the patches are sampled, you can assign the function to compute the
            texture features using this setting. This should be used with ``patch_sampling_callback_kwargs`` and also
            ``create_new_attribute``. Default to ``None``.
        patch_sampling_callback_kwargs (dict):
            The kwargs that will be supplied to the lambda function specified by ``patch_sampling_callback``.
            Default to empty dict.
        create_new_attribute (str):
            The new data created by `patch_sampling_callback` will be attached to the subject using this argument
            as the attribute name. The new data can then be accessed by ``tio.Subject()[create_new_attribute]``.
            Default to ``None``.
        inf_samples_per_vol (int):
            Sometimes inference will require more sampled patches to generate appealing results (e.g, segmentation),
            this argument,

    """
    data_types                    : Iterable = [float, float]
    sampler                       : str      = 'weighted'           # 'weighted' or 'uniform'
    sampler_kwargs                : dict     = dict()               # pass to ``tio.Sampler``
    augmentation                  : str      = None                 # yaml file to create tio transform
    create_new_attribute          : str      = None                 # create a new attribute in subjects for callback
    patch_sampling_callback       : Callable = None                 # callback to generate new data
    patch_sampling_callback_kwargs: dict     = dict()               # kwargs pass to the callback
    inf_samples_per_vol           : int      = None                 # number of samples per volume during inference
    mask_dir                      : str      = None                 # Image dir used by ``tio.Sampler``
    probmap_dir                   : str      = None                 # Image dir used by ``tio.WeightedSampler``
    tio_queue_kwargs = dict(            # dict passed to ``tio.Queue``
        max_length             = 15,
        samples_per_volume     = 1,
        num_workers            = 16,
        shuffle_subjects       = True,
        shuffle_patches        = True,
        start_background       = True,
        verbose                = False,
    )


class PMIImageDataLoader(PMIDataLoaderBase):
    """
    This class load :class:ImageDataSet related image data. Customization of loading this class should inherit this
    class.

    This class is suitable in the following situations:
        * Image to image
        * Image to segmentation
        * Image super-resolution

    Attributes:
        Attributes will be loaded from the supplied ``cfg`` into class

    Args:
        *args: Please see parent class.
        **kwargs: Please see parent class.

    .. note::
        Attributes are defined in :func:`PMIImageDataLoader._read_config`, either read from a dictionary or an ini
        file. The current system reads the [LoaderParams].

    .. hint::
        Users are suppose to pass arguments to the super class for handling. If in doubt, look at the docs of parent
        class!


    See Also:
        :class:`PMIDataLoaderBase`
    """
    def __init__(self, cfg: PMIImageDataLoaderCFG, *args, **kwargs):
        super(PMIImageDataLoader, self).__init__(cfg, *args, **kwargs)

    def _check_input(self):
        """Not implemented."""
        return True

    def _read_config(self, config_file=None):
        """
        Defines attributes. Called when object is created. Extra attributes are declared in super function,
        see the super class for more details. Params are read from `[LoaderParams]` section of the ini.

        Args:
            config_file (str or dict, Optional): See :func:`PMIDataLoaderBase._read_config`.

        See Also:
            * :class:`PMIDataLoaderBase`
            * :func:`PMIDataLoaderBase._read_config`
        """
        super(PMIImageDataLoader, self)._read_config(config_file)

        # Update some kwargs with more complex default settings
        default_queue_kwargs = self._cfg.tio_queue_kwargs.copy()
        if default_queue_kwargs['num_workers'] > mpi.cpu_count():
            default_queue_kwargs['num_workers'] = mpi.cpu_count()
        self.tio_queue_kwargs = default_queue_kwargs

        # If samplers are specified create tio queues using these samplers.
        if (self.sampler == 'weighted') :
            if self.probmap_dir is None:
                msg = f"Weighted samplers requires probability map to sample patches. Specify 'probmap_dir' in cfg. "
                raise KeyError(msg)
            else:
                # Default attribute for probability map is 'probmap'
                if not 'probability_map' in self.sampler_kwargs:
                    self.sampler_kwargs['probability_map'] = 'probmap'
            if not set(['patch_size', 'probability_map']).issubset(set(self.sampler_kwargs.keys())):
                msg = f"`sampler_kwargs` must contain both 'patch_size' and 'probability_map' keys for weighted" \
                      f"samplers. Got {self.sampler_kwargs} instead."
                raise KeyError(msg)
            self.sampler_instance = tio.WeightedSampler(**self.sampler_kwargs)
        elif (self.sampler == 'uniform'):
            if not 'patch_size' in self.sampler_kwargs:
                msg = f"Require 'patch_size' argument to use ``tio.UniformSampler``. Specify 'patch_size' in ``cfg.samp" \
                      f"ler_kwargs' dictionary."
                raise KeyError(msg)
            self.sampler_instance = tio.UniformSampler(**self.sampler_kwargs)
        elif (self.sampler == 'grid'):
            if not 'patch_size' in self.sampler_kwargs:
                msg = f"Require 'patch_size' argument to use ``tio.GridSampler``. Specify 'patch_size' in ``cfg.samp" \
                      f"ler_kwargs' dictionary."
                raise KeyError(msg)
            if not 'patch_overlap' in self.sampler_kwargs:
                # Automatically determine overlap if its not specified
                overlap = [ps // 2 for ps in self.sampler_kwargs['patch_size']]
                self.sampler_kwargs['patch_overlap'] = overlap
            self.sampler_instance = tio.GridSampler(**self.sampler_kwargs)
        else:
            # If sampler is not specified, assume the whole image is sampled
            self.sampler_instance = None
        self.queue_args = [self.tio_queue_kwargs.pop(k)
                           for k in ['max_length', 'samples_per_volume']] \
                          + [self.sampler_instance] # follows torchio's args arrangments

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
                                      regex=self.id_globber, idlist=self.id_list, idGlobber=self.id_globber, **kwargs)
        return img_data

    def _load_data_set_training(self,
                                exclude_augment: bool = False) -> tio.Queue:
        """
        Load ImageDataSet for input and segmentation. For more see :func:`create_transform()`.
        """
        if self.target_dir is None:
            raise IOError(f"Cannot load from {self.target_dir}")

        img_out = self._read_image(self.input_dir, dtype=self.data_types[0])
        gt_out = self._read_image(self.target_dir, dtype=self.data_types[1])
        mask_out = self._read_image(self.mask_dir, dtype='uint8')
        prob_out = self._prepare_probmap()

        self.data = self._prepare_data(gt_out, img_out, mask_out, prob_out)
        # Create transform
        transform = self._create_transform(exclude_augment=exclude_augment)

        # Create subjects & queue
        subjects = self._pack_data_into_subjects(self.data, transform)

        # Return the queue
        return self._create_queue(exclude_augment, subjects)

    def _load_data_set_inference(self) -> [tio.Queue, tio.GridSampler] or [tio.SubjectsDataset, None]:
        """Same as :func:`_load_data_set_training` in this class, except the ground-truth is
        not loaded."""
        img_out = self._read_image(self.input_dir, dtype=self.data_types[0])
        prob_out = self._prepare_probmap()

        if not self.target_dir in (None, 'None'):
            try:
                gt_out = self._read_image(self.target_dir, dtype=self.data_types[1])
            except:
                self._logger.exception("Can't load from: {}".format(self.target_dir))
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
        return self._create_queue(True, subjects, return_sampler=False, training=False)

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
        if self.probmap_dir is not None:
            self._logger.info(f"Loading probmap from: {self.probmap_dir}")
            try:
                prob_out = self._read_image(self.probmap_dir, dtype='uint32') # torchio requires Integer probmap
            except:
                self._logger.warning(f"Couldn't open probmap from: {self.probmap_dir}", no_repeat=True)
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
                      training: Optional[bool]=False,
                      return_sampler: Optional[bool]=False) -> [tio.Queue, tio.GridSampler] or \
                                                                [tio.SubjectsDataset, None]:
        r"""This method build the queue from the input subjects. If the queue involves a :class:`tio.GridSampler`,
        it is generally needed by the inferencer to reconstruct it back into its original dimension. Thus, an
        optional to also return the sampler is given.

        Args:
            exclude_augment (bool):
                If true, ignore all augmentation transform in ``self.transform``.
            subjects (tio.SubjectsDataset):
                Subjects to be loaded into queue.
            training (bool, Optional):
                ``True`` = training mode. ``False`` = inference mode.
            return_sampler (bool, Optional):
                If ``True``, return the ``tio.Sampler`` alongside the ``tio.Queue``. This is useful during inference
                where you need to keep the ``tio.Sampler`` to create the aggregator that will assemble the patches.
        """
        # default is based on self._training_mode, read from config file
        if training is None:
            training = self._run_mode

        if self.sampler_instance is None:
            # Set queue_args and queue_kwargs to load the whole image for each object to allow for caching
            shape_of_input = subjects[0].shape

            # Reset sampler
            self.sampler_instance = tio.UniformSampler(patch_size=shape_of_input[1:])  # first dim is batch
            self.queue_args[-1] = self.sampler_instance

        queue_dict = self.tio_queue_kwargs.copy()
        # if exclude augment, don't shuffle
        if exclude_augment:
            queue_dict['shuffle_subjects'] = False
        if not training:
            # don't shuffle subject if inference
            queue_dict['shuffle_subjects'] = False
            queue_dict['shuffle_patches'] = False # This might haunt you later because some inference might require
                                                  # shuffling the patches (e.g., grid sampler)

        # Create queue
        # If option to use post-sampling processing was provided, use CallbackQueue instead
        if  not self.patch_sampling_callback in ("", None):
            # check if there's illegal characters in the patch_sampling_callback
            if re.search("[\W]+", self.patch_sampling_callback.translate(str.maketrans('', '', "[], "))) is not None:
                raise AttributeError(f"You patch_sampling_callback specified ({self.patch_sampling_callback}) "
                                     f"contains illegal characters!")
            queue_dict['start_background'] = training # if not training, delay dataloading
            _callback_func = eval(self.patch_sampling_callback)
            _callback_func = partial(_callback_func, **self.patch_sampling_callback_kwargs)
            queue = CallbackQueue(subjects, *self.queue_args,
                                  patch_sampling_callback=_callback_func,
                                  create_new_attribute = self.create_new_attribute,
                                  **queue_dict)
        else: # Else use the normal queue
            # queue_dict.pop('patch_sampling_callback')
            # queue_dict.pop('create_new_attribute')
            # ignore 'start_background` option for ordinary queues
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

        if isinstance(self.sampler_instance, tio.GridSampler):
            self.sampler_instance.set_subject(subject)
        elif isinstance(self.sampler_instance, tio.WeightedSampler):
            _spv = self.inf_samples_per_vol if self.inf_samples_per_vol is not None \
                else self.queue.samples_per_volume
            self._logger.info(f"Setting the number of patches to sample to: "
                              f"{_spv}")
            self.sampler_instance.set_subject(subject, _spv)
            self.queue.samples_per_volume = _spv
        else:
            msg = f"Currrently only support GridSampler and WeightedSampler, but got {type(self.sampler_instance)}"
            raise TypeError(msg)

        # Replace subjects in queue and reset the queue
        if isinstance(subject, tio.Subject):
            subject = tio.SubjectsDataset([subject])
        del self.queue._subjects_iterable, self.queue.sampler
        self.queue.subjects_dataset = subject
        self.queue._subjects_iterable = None
        self.queue.sampler = self.sampler_instance
        self.queue._initialize_subjects_iterable()
        aggregator = tio.GridAggregator(self.sampler_instance, 'average')
        return self.queue, aggregator

    def get_sampler(self):
        return self.sampler_instance