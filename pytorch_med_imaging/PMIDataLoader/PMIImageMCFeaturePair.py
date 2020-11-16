from .PMIImageDataLoader import PMIImageDataLoader
from .. import MedImgDataset

__all__ = ['PMIImageMCFeaturePair']

class PMIImageMCFeaturePair(PMIImageDataLoader):
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
            root_dir (str): See :class:`MedImgDataset.ImageDataSet`
            **kwargs: See :class:`MedImgDataset.ImageDataSet`

        Raises:
            AttributeError: If there are no corresponding items in section `[LoaderParams]`.

        Returns:
            (ImageDataSet or ImageDataSetAugment): Loaded image data set.

        See Also:
            :class:`MedImgDataset.ImageDataSet`
        """
        # default reader func
        self._image_class = MedImgDataset.ImageDataMultiChannel

        return self._image_class(root_dir, channel_subdirs=self._channel_subdirs, verbose=self._verbose,
                                 debugmode=self._debug, filtermode='both', regex=self._regex, idlist=self._idlist,
                                 loadBySlices=self._load_by_slices, aug_factor=self._augmentation, **kwargs)

    def _load_data_set_training(self):
        """
        Load :class:`ImageDataSet` or :class:`ImageDataSetAugment for network input.
        Load :class:`DataLabel` as target.

        Returns:
            (tuple) -> (:class:`ImageDataSet` or :class:`ImageDataSetAugment`, :class:`DataLabel`)

        """
        img_out = self._read_image(self._input_dir)

        if not self.get_from_config('excel_sheetname', None) is None:
            gt_dat = MedImgDataset.DataLabel.from_xlsx(self._target_dir, self.get_from_config('excel_sheetname', None))
        else:
            gt_dat = MedImgDataset.DataLabel.from_csv(self._target_dir)

        # Load selected columns only
        if not self.get_from_loader_params('column') is None:
            self._logger.info("Selecting target column: {}".format(self.get_from_loader_params('column')))
            gt_dat.set_target_column(self.get_from_loader_params('column'))

        gt_dat.map_to_data(img_out)
        return img_out, gt_dat

