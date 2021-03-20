from .PMIImageDataLoader import PMIImageDataLoader
from ..med_img_dataset import ImagePatchesLoader
from ..med_img_dataset.computations import clip_5, ImagePatchLocMMTex

__all__ = ['PMIImagePatchesLoader']

class PMIImagePatchesLoader(PMIImageDataLoader):
    """
    Load .nii.gz files as 2D patches. This class uses the :class:`ImagePatchesLaoder` class.

    A list of parameters that is read by this class:

    * 'data_type`
    * `regex`
    * `idlsit`
    * `augmentation`
    * `load_by_slice`
    * `patch_size`
    * `patch_stride`
    * `include_last_patch`
    * `random_patches`
    * `renew_index`
    * `axis`
    * `random_from_distribution`

    If additional options `compute_textures` is provide, the class :class:`ImagePatchLocTex` will be used instead.
    The value of `load_textures` should be either: `{'as_channels'|'as_histograms'}`. Anything else, it will
    retreat to using 'as_channels' unless its `None`.

    For usage, please see See Also

    .. note::
        Attributes list here in stores in different variables and are not grouped.

    See Also:
        * :class:`med_img_dataset.ImagePatchesLoader`
        * :class:`PMIImageDataLoader`

    """
    def __init__(self, *args, **kwargs):
        super(PMIImagePatchesLoader, self).__init__(*args, **kwargs)


    def _read_params(self, config_file=None):
        """
        Additional attributes are read for :class:`med_img_dataset.ImagePatchesLoader`. To specify these attributes,
        you can add a section `[LoaderParams]` in the .ini config files.

        For attributes read, refer to class descriptions.

        See Also:
            :class:`med_img_dataset.ImagePatchesLoader`
        """
        super(PMIImagePatchesLoader, self)._read_params(config_file)

        keys = ['patch_size',
                'patch_stride',
                'include_last_patch',
                'random_patches',
                'renew_index',
                'random_from_distribution',
                'compute_textures'
                ]

        default_values = [128,
                          32,
                          False,
                          0,
                          True,
                          clip_5,
                          None]

        # Note that 'compute_textures' should be string.
        eval_flags = [True] * (len(keys) - 1) + [False]

        self._patch_loader_params = self.get_target_attributes('LoaderParams', keys, default_values, eval_flags)
        self._patch_size = self._patch_loader_params.pop('patch_size')

        self._logger.info(f"Params read: {self._patch_loader_params}")


    def _load_data_set_training(self):
        """
        Read as :class:`med_img_dataset.ImageDataSet` or :class:`med_img_dataset.ImageDataSetAugment` first, then uses
        :class:`med_img_dataset.ImagePatchesLoader` to sample patches from them.

        Returns:
            (list of :class:`MedImgDataset.ImagePatchesLoader`): Input to network and to loss function respectively.
        """
        img_out, gt_out = super(PMIImagePatchesLoader, self)._load_data_set_training()

        if self._patch_loader_params['compute_textures'] is not None:
            data_cls = ImagePatchLocMMTex
            self._patch_loader_params['mode'] = self._patch_loader_params.pop('compute_textures')
        else:
            data_cls = ImagePatchesLoader
        self._logger.info(f"Using data class: {data_cls.__name__}")

        img_out = data_cls(img_out, self._patch_size, pre_shuffle=False, **self._patch_loader_params)
        seg_out = ImagePatchesLoader(gt_out, self._patch_size, pre_shuffle=False, reference_dataset=img_out, **self._patch_loader_params)

        return img_out, seg_out

    def _load_data_set_inference(self):
        """
        Read as :class:`med_img_dataset.ImageDataSet` or :class:`med_img_dataset.ImageDataSetAugment` first, then uses
        :class:`med_img_dataset.ImagePatchesLoader` to sample patches from them.

        Returns:
            (:class:`MedImgDataset.ImagePatchesLoader`): Input to network.
        """
        img_out = super(PMIImagePatchesLoader, self)._load_data_set_inference()

        if self._patch_loader_params['compute_textures'] is not None:
            data_cls = ImagePatchLocMMTex
            self._patch_loader_params['mode'] = self._patch_loader_params.pop('compute_textures')
        else:
            data_cls = ImagePatchesLoader

        # Reqruiements from dataclass, force this to 0 if in inference mode.
        self._patch_loader_params['random_patches'] = 0

        img_out = data_cls(img_out, self._patch_size, pre_shuffle=True, **self._patch_loader_params)
        return img_out