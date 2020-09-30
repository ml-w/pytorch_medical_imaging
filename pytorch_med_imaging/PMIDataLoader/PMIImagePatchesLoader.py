from .PMIImageDataLoader import PMIImageDataLoader
from .. import MedImgDataset


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

    For usage, please see See Also

    .. note::
        Attributes list here in stores in different variables and are not grouped.

    See Also:
        * :class:`MedImgDataset.ImagePatchesLoader`
        * :class:`PMIImageDataLoader`

    """
    def __init__(self, *args, **kwargs):
        super(PMIImagePatchesLoader, self).__init__(*args, **kwargs)


    def _read_params(self, config_file=None):
        """
        Additional attributes are read for :class:`MedImgDataset.ImagePatchesLoader`. To specify these attributes,
        you can add a section `[LoaderParams]` in the .ini config files.

        For attributes read, refer to class descriptions.

        See Also:
            :class:`MedImgDataset.ImagePatchesLoader`
        """
        super(PMIImagePatchesLoader, self)._read_params(config_file)

        keys = ['patch_size',
                'patch_stride',
                'include_last_patch',
                'random_patches',
                'renew_index',
                'axis',
                'random_from_distribution'
                ]

        default_values = [0,
                          -1,
                          False,
                          0,
                          False,
                          None,
                          None]

        eval_flags = [True] * len(keys)

        self._patch_loader_params = self.get_target_attributes(keys, default_values, eval_flags)
        self._patch_size = self._patch_loader_params['patch_size']


    def _load_data_set_training(self):
        """
        Read as :class:`MedImgDataset.ImageDataSet` or :class:`MedImgDataset.ImageDataSetAugment` first, then uses
        :class:`MedImgDataset.ImagePatchesLoader` to sample patches from them.

        Returns:
            (list of :class:`MedImgDataset.ImagePatchesLoader`): Input to network and to loss function respectively.
        """
        img_out, gt_out = super(PMIImagePatchesLoader, self)._load_data_set_training()

        img_out = MedImgDataset.ImagePatchesLoader(img_out, self._patch_size, pre_shuffle=True, **self._patch_loader_params)
        seg_out = MedImgDataset.ImagePatchesLoader(gt_out, self._patch_size, pre_shuffle=True, reference_dataset=img_out)

        return img_out, seg_out

    def _load_data_set_inference(self):
        """
        Read as :class:`MedImgDataset.ImageDataSet` or :class:`MedImgDataset.ImageDataSetAugment` first, then uses
        :class:`MedImgDataset.ImagePatchesLoader` to sample patches from them.

        Returns:
            (:class:`MedImgDataset.ImagePatchesLoader`): Input to network.
        """
        img_out = super(PMIImagePatchesLoader, self)._load_data_set_inference()
        img_out = MedImgDataset.ImagePatchesLoader(img_out, self._patch_size, pre_shuffle=False **self._patch_loader_params)
        return img_out