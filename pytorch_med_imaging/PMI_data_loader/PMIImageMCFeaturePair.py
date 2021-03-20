from .PMIImageFeaturePair import PMIImageFeaturePair
from .. import med_img_dataset

__all__ = ['PMIImageMCFeaturePair']

class PMIImageMCFeaturePair(PMIImageFeaturePair):
    """
    This class load :class:`ImageDataMultiChannel` related image data together with features written in a csv file.

    This class is suitable in the following situations:
        * Image to features prediction
        * Image classifications
        * Image to coordinates

    Attributes:
        regex (str, Optional):
            Filter for loading files. See :class:`ImageDataSet`
        idlist (str or list, Optional):
            Filter for loading files. See :class:`ImageDataSet`
        augmentation (int):
            If `_augmentation` > 0, :class:`ImageDataSetAugment` will be used instead.
        load_by_slice (int):
            If `_load_by_slice` > -1, images volumes are loaded slice by slice along the axis specified.
        concat_by_axis (int):
            If `concat_by_axis` > -1, images volume are concatenated at the specified axis instead of by channels.

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
        super(PMIImageMCFeaturePair, self).__init__(*args, **kwargs)

    def _read_params(self, config_file=None):
        super(PMIImageMCFeaturePair, self)._read_params(config_file)

        # Currently not support augmentation
        self._augmentation = 0

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
                                 debugmode=self._debug, filtermode='both', regex=self._regex, idlist=self._idlist,
                                 loadBySlices=self._load_by_slices, aug_factor=self._augmentation,
                                 concat_by_axis=concat_by_axis, **kwargs)



