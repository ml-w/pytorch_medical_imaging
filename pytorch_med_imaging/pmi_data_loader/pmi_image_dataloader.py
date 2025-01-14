import os
import torchio
from .pmi_dataloader_base import PMIDataLoaderBase, PMIDataLoaderBaseCFG
from .. import med_img_dataset
from ..med_img_dataset import ImageDataSet
from .lambda_tio_adaptor import CallbackQueue
from .computations.queue_callback import *
from typing import *
from functools import partial
from dataclasses import dataclass
import torchio as tio
import re

__all__ = ['PMIImageDataLoader', 'PMIImageDataLoaderCFG']

class PMIImageDataLoaderCFG(PMIDataLoaderBaseCFG):
    r"""Configuration for :class:`PMIImageDataLoader`.

    Class Attributes:
        data_types (iterable, Optional):
            Data type of input and ground-truth. Depending on how you use the dataloader, but generally speaking, its
            the desired data type for input data and target data. Default to ``[float, float]``.
        sampler (str, Optional):
            Determine the ``tio.Sampler`` used to sample the images. Support ['weighted'|'uniform'|'grid'] currently.
            Default to ``None``, which means no sampler is used (i.e., the whole image is loaded).
        sampler_kwargs (dict, Optional):
            The kwargs passed to ``tio.Sampler``. For 'weighted', key ``patch_size`` and ``prob_map`` is required. For
            'uniform', only ``patch_size`` is required. Unless sampler is ``None``, this needs to be specified. Default
            value is only a place holder.
        augmentation (str, Optional):
            Path to yaml file to create the ``tio.Compose`` transform. Default to ``None``.
        patch_sampling_callback (Callable or str, Optional):
            A function that is called after patch sampling to generate new data from sampled patches. For example, if
            texture features is required after the patches are sampled, you can assign the function to compute the
            texture features using this setting. This should be used with ``patch_sampling_callback_kwargs`` and also
            ``create_new_attribute``. Default to ``None``.
        patch_sampling_callback_kwargs (dict, Optional):
            The kwargs that will be supplied to the lambda function specified by ``patch_sampling_callback``.
            Default to empty dict.
        create_new_attribute (str, Optional):
            The new data created by `patch_sampling_callback` will be attached to the subject using this argument
            as the attribute name. The new data can then be accessed by ``tio.Subject()[create_new_attribute]``.
            Default to ``None``.
        inf_samples_per_vol (int, Optional):
            Sometimes inference will require more sampled patches to generate appealing results (e.g, segmentation),
            this argument,

    .. hint::
        **Minimal required attributes:**

        * :attr:`input_dir` - Load image to subject key `'input'`.
        * :attr:`target_dir` - Load target to subject key `'gt'`

        **Default subject packaging keys:**

        * ``['input', 'gt', 'mask', 'prob_map', 'uid']``. See :func:`PMIImageDataLoader._prepare_data` for the details.


    See Also:
        * :class:`PMIDataLoaderBaseCFG`

    """
    data_types                    : Optional[Iterable] = [float, float]
    sampler                       : Optional[str]      = None                 # 'weighted' or 'uniform'
    sampler_kwargs                : Optional[dict]     = dict()               # pass to ``tio.Sampler``
    augmentation                  : Optional[str]      = None                 # yaml file to create tio transform
    force_augment                 : Optional[bool]     = False
    create_new_attribute          : Optional[str]      = None                 # create a new attribute in subjects for callback
    patch_sampling_callback       : Optional[Callable] = None                 # callback to generate new data
    patch_sampling_callback_kwargs: Optional[dict]     = dict()               # kwargs pass to the callback
    inf_samples_per_vol           : Optional[int]      = None                 # number of samples per volume during inference
    mask_dir                      : Optional[str]      = None                 # Image dir used by ``tio.Sampler``
    probmap_dir                   : Optional[str]      = None                 # Image dir used by ``tio.WeightedSampler``
    tio_queue_kwargs              : Optional[dict]     = dict(            # dict passed to ``tio.Queue``
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

    Class Attributes:
        cfg_cls (type):
            The class of the CFG that is default for this loader.

    Args:
        *args: Please see parent class.
        **kwargs: Please see parent class.

    .. note::
        Attributes are defined in :class:`PMIImageDataLoaderCFG`.

    .. hint::
        Users are suppose to pass arguments to the super class for handling. If in doubt, look at the docs of parent
        class!


    See Also:
        :class:`PMIDataLoaderBase`
    """
    cfg_cls = PMIImageDataLoaderCFG
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
        if default_queue_kwargs['num_workers'] > os.cpu_count():
            default_queue_kwargs['num_workers'] = os.cpu_count()
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
        """Private method for convenience.

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
        if root_dir is None or root_dir == '':
            self._logger.warning("Received `None` for root_dir arguement.")
            return None

        self._image_class = med_img_dataset.ImageDataSet
        img_data =  self._image_class(root_dir, debugmode=self.debug_mode, filtermode='both',
                                      regex=self.id_globber, idlist=self.id_list, id_globber=self.id_globber, **kwargs)
        return img_data

    def _load_data_set_training(self,
                                exclude_augment: bool = False) -> tio.Queue:
        """
        Load ImageDataSet for input and segmentation. For more see :func:`create_transform()`.

        Returns:
            tio.Queue
        """
        if self.target_dir is None:
            raise IOError(f"Cannot load from {self.target_dir}")

        self.data = self._prepare_data()
        # Create transform
        transform = self._create_transform(exclude_augment=exclude_augment)

        # Create subjects & queue
        subjects = self._pack_data_into_subjects(self.data, transform)

        # Return the queue
        return self._create_queue(exclude_augment, subjects)

    def _load_gt_data(self) -> ImageDataSet or None:
        r"""For inheritance completeness

        Returns:
            ImageDataSet or None
        """
        if not self.target_dir is None:
            self._logger.info(f"Reading gt data from {self.target_dir}")
            return self._read_image(self.target_dir, dtype=self.data_types[1])
        else:
            self._logger.exception("Can't load from: {}".format(self.target_dir))
            self._logger.warning("Skipping ground-truth data loading.")
            return None

    def _load_data_set_inference(self) -> [tio.Queue, tio.GridSampler] or [tio.SubjectsDataset, None]:
        """Same as :func:`_load_data_set_training` in this class, except the ground-truth is optional to load and
        the transform will ignore augmentation
        """
        # override the number of patches drawn for inference if this option is provided
        if self.inf_samples_per_vol is not None:
            self._logger.info(f"Override `samples_per_vol` {self.queue_args[1]} with "
                              f"`inf_samples_per_vol` {self.inf_samples_per_vol}")
            self.queue_args[1] = int(self.inf_samples_per_vol)

        # set ``exclude_augment`` to False for normal inference unless `force_augment
        if self.force_augment:
            self._logger.warning(f"Force data augmentation during inference.")
        return self._load_data_set_training(exclude_augment=not self.force_augment)


    def _prepare_data(self) -> dict:
        """This is an important function that will prepare the data as a dictionary. This dictionary will be passed
        to :func:`_pack_data_into_subjects` and then :func:`_create_queue`. The queue will return a ``dict`` like
        instance, which is characterized by the keys specified in this function.

        .. hint::
            Override this function in the child classes to alter the behavior of data loading. You might also want
            to override :func:`_load_gt_data` and, depend on circumstances, :func:`_load_data_set_inference` too.

        Args:
            gt_out   (Any)         : Target data.
            img_out  (ImageDataSet): Input data.
            mask_out (ImageDataSet): Referenced by ``tio.Compose`` during transform.
            prob_out (ImageDataSet): Referenced by ``tio.Sampler`` during sampling.
        """
        self._logger.info("Reading input images...")
        img_out = self._read_image(self.input_dir, dtype=self.data_types[0])
        gt_out = self._load_gt_data()
        self._logger.info("Reading masks...")
        mask_out = self._read_image(self.mask_dir, dtype='uint8')
        prob_out = self._prepare_probmap()

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

    def _create_queue(self,
                      exclude_augment: bool,
                      subjects: tio.SubjectsDataset,
                      training: Optional[bool]=None,
                      return_sampler: Optional[bool]=False) -> [tio.Queue, tio.GridSampler] or \
                                                                [tio.SubjectsDataset, None]:
        r"""This method build the queue from the input subjects. If the queue involves a :class:`tio.GridSampler`,
        it is generally needed by the inferencer to reconstruct it back into its original dimension. Thus, an
        optional to also return the sampler is given. If :attr:`sampler_instance` is ``None``, ``tio.UniformSampler``
        will be used to sample 1 patch with the size defined by the image shape of the first subject in ``subjects``.

        Args:
            exclude_augment (bool):
                If true, ignore all augmentation transform in ``self.transform``.
            subjects (tio.SubjectsDataset):
                Subjects to be loaded into queue.
            training (bool, Optional):
                ``True`` = training mode. ``False`` = inference mode. If ``None``, respect ``self.run_mode``.
            return_sampler (bool, Optional):
                If ``True``, return the ``tio.Sampler`` alongside the ``tio.Queue``. This is useful during inference
                where you need to keep the ``tio.Sampler`` to create the aggregator that will assemble the patches.
        """
        # default is based on self._training_mode, read from config file
        if training is None:
            training = self.run_mode

        queue_dict, training = self._prepare_queue_dict(exclude_augment, subjects, training)
        if self.sampler is not None:
            # Create queue
            # If option to use post-sampling processing was provided, use CallbackQueue instead
            if  not self.patch_sampling_callback in ("", None):
                if isinstance(self.patch_sampling_callback, str):
                    # check if there's illegal characters in the patch_sampling_callback
                    if re.search("[\W]+", self.patch_sampling_callback.translate(str.maketrans('', '', "[], "))) is not None:
                        raise AttributeError(f"You patch_sampling_callback specified ({self.patch_sampling_callback}) "
                                             f"contains illegal characters!")
                    _callback_func = eval(self.patch_sampling_callback)
                elif callable(self.patch_sampling_callback):
                    self._logger.info("Detect callable patch sampling callback specified.")
                    _callback_func = self.patch_sampling_callback
                    pass

                _callback_func = partial(_callback_func, **self.patch_sampling_callback_kwargs)
                queue_dict['start_background'] = training # if not training, delay dataloading
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
        else:
            # Because no sampler means the samples are loaded by default dataloader, it is needed to change the
            # ``torch.utils.data.DataLoader` parameters to allow parallel loading. However, this is done in the
            # method ``get_torch_data_loader()``.
            # queue = subjects
            self._logger.debug(f"No sampler specified, use first shape in subjects: {subjects[0].shape[1:]}.")
            # Ad crop-or-pad to prevent shape issues
            crop_or_pad = tio.CropOrPad(target_shape=subjects[0].shape[1:])
            if isinstance(subjects._transform, tio.Compose):
                subjects._transform.transforms.append(crop_or_pad)
            elif isinstance(subjects._transform, tio.Transform):
                subjects._transform = tio.Compose([subjects._transform, crop_or_pad])
            else:
                subjects.set_transform(crop_or_pad)
            _sampler = tio.UniformSampler(patch_size = subjects[0].shape[1:])
            queue = tio.Queue(subjects, sampler=_sampler, samples_per_volume=1, max_length=self.queue_args[0],
                              **queue_dict)
            self._logger.debug(f"Created queue: {queue}")
            self.queue = queue
        if return_sampler and self.sampler is not None:
            return queue, self.queue_args[-1]
        else:
            return queue

    def _prepare_queue_dict(self, exclude_augment, subjects, training) -> [dict, bool]:
        r"""Rearrange some of the tags to cater for differences in the need to shuffle subjects and patches. If
        :attr:`sampler_instance` is ``None``, ``tio.UniformSampler`` will be used to sample 1 patch with the size
        defined by the image shape of the first subject in ``subjects``.

        See Also:
            * :func:`_create_queue`

        Return:
            tuple(dict, bool)
        """
        if training is None:
            training = self.run_mode
        if self.sampler_instance is None:
            self._logger.info("No sampler is specified, setting the sampler to uniform and patchsize to shape of "
                              "first image.")
            # Set queue_args and queue_kwargs to load the whole image for each object to allow for caching
            shape_of_input = subjects[0].shape
            self._logger.debug(f"Shape of first image: {shape_of_input}")

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
            queue_dict['shuffle_patches'] = False  # This might haunt you later because some inference might require
            # shuffling the patches (e.g., grid sampler)
        return queue_dict, training

    def create_aggregation_queue(self, subjects: torchio.SubjectsDataset, *args, **kwargs):
        r"""This method samples from the input subjects and return the queue and also the aggregator. The sampled
        patches is obtained by iterating the updated ``self.queue``, which are supposed to be processed by the network.
        The processed output should have the same dimension as the input (except for channel), and are fed into the
        aggregator produced by this method.

        .. important::
            Note that this function should only be invoked during inference. Typically, you don't need the
            aggregator anywhere else.

        Args:
            subjects (SubjectDataset):
                The target subject to sample from using the configured sampler.

        Returns:
            [torchio.Queue, torchio.GridAggregator]
        """
        required_att = ('sampler', 'data', 'queue')
        for att in required_att:
            if not hasattr(self, att):
                msg += f"Attribute {att} missing. Have you run load_data() already?"
                raise AttributeError(msg)

        if isinstance(self.sampler_instance, tio.GridSampler):
            self.sampler_instance.set_subject(subjects)
        elif isinstance(self.sampler_instance, tio.WeightedSampler):
            _spv = self.inf_samples_per_vol if self.inf_samples_per_vol is not None \
                else self.queue.samples_per_volume
            self._logger.info(f"Setting the number of patches to sample to: "
                              f"{_spv}")
            self.sampler_instance.set_subject(subjects, _spv)
            self.queue.samples_per_volume = _spv
        else:
            msg = f"Currrently only support GridSampler and WeightedSampler, but got {type(self.sampler_instance)}"
            raise TypeError(msg)

        # Replace subjects in queue and reset the queue
        if isinstance(subjects, tio.Subject):
            subjects = tio.SubjectsDataset([subjects])
        del self.queue._subjects_iterable, self.queue.sampler
        self.queue.subjects_dataset = subjects
        self.queue._subjects_iterable = None
        self.queue.sampler = self.sampler_instance
        self.queue._initialize_subjects_iterable()
        aggregator = tio.GridAggregator(self.sampler_instance, 'average')
        return self.queue, aggregator

    def get_sampler(self):
        return self.sampler_instance

    def get_inf_samples_per_vol(self):
        r"""Sometimes you want more samples per volume for weighted samplers. You can directly do it by accessing the
        attribute :attr:`inf_samples_per_vol`, but it is unsafe as it is not always defined. This method is a safe
        method to get that information.

        Returns:
            int: Samples per volume for inference
        """
        f = getattr(self, 'inf_samples_per_vol', None)
        if f is None:
            f = getattr(self.queue, 'samples_per_volume', None)
        if f is None:
            msg = "Cannot get inf_samples_per_volume."
            raise AttributeError(msg)

    @property
    def patch_size(self):
        r"""Return the patch_size passed to tio_queue_kwargs"""
        if hasattr(self.sampler_kwargs, 'patch_size'):
            if self.sampler_instance is None:
                msg = f"Trying to fetch patch_size before sampler instance creation. This might lead to the deviation "\
                      f"of the actual sample patch_size and the value returned by this function."
                self._logger.warning(msg)
            return self.sampler_kwargs.patch_size
        else:
            msg = "Trying to fetch patch_size but it was not defined."
            self._logger.warning(msg)
            return None

    @patch_size.setter
    def patch_size(self, patch_size):
        r"""Convinient method to set :attr:`.patch_size`

        Args:
            patch_size (Iterable[int]):
                Desired patch size for sampler. Only works when :attr:`.sampler_instance` is created. The input should
                be in the order :math:`(W × H × D)`.
        """
        self.sampler_kwargs['patch_size'] = patch_size