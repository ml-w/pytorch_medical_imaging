from .pmi_img_feat_pair_dataloader import PMIImageFeaturePairLoader
from .. import med_img_dataset

__all__ = ['PMIImageMCFeaturePairLoader']


class PMIImageMCFeaturePairLoader(PMIImageFeaturePairLoader):
    r"""
    This class load :class:`ImageDataMultiChannel` related image data together with features written in a csv file.

    This class is suitable in the following situations:
        * Image to features prediction
        * Image classifications
        * Image to coordinates

    Args:
        *args: Please see parent class.
        **kwargs: Please see parent class.

    .. note::


    .. hint::
        Users are suppose to pass arguments to the super class for handling. If in doubt, look at the docs of parent
        class!


    See Also:
        :class:`PMIDataLoaderBase`

    """
    def __init__(self, *args, **kwargs):
        super(PMIImageMCFeaturePairLoader, self).__init__(*args, **kwargs)

    def _read_config(self, config_file=None):
        super(PMIImageMCFeaturePairLoader, self)._read_config(config_file)

        # Load if subdirs are specified
        self._channel_subdirs = self.get_from_loader_params_with_eval('channel_subdirs', None)
        assert isinstance(self._channel_subdirs, list), \
            "Specified channel_subdir is not a list: {}".format(self._channel_subdirs)


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
        # default reader func
        self._image_class = med_img_dataset.ImageDataMultiChannel

        concat_by_axis = self.get_from_loader_params_with_eval('concat_by_axis', -1)
        return self._image_class(root_dir, channel_subdirs=self._channel_subdirs, verbose=self._verbose,
                                 debugmode=self.debug_mode, filtermode='both', regex=self.id_globber, idlist=self.id_list,
                                 loadBySlices=self._load_by_slices, aug_factor=self._augmentation,
                                 concat_by_axis=concat_by_axis, **kwargs)



