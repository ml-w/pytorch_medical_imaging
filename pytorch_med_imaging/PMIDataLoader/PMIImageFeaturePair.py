from .PMIImageDataLoader import PMIImageDataLoader
from .. import MedImgDataset

__all__ = ['PMIImageFeaturePair']

class PMIImageFeaturePair(PMIImageDataLoader):
    """
    This class load :class:ImageDataSet related image data together with features written in a csv file.

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
        super(PMIImageDataLoader, self).__init__(*args, **kwargs)


    def _load_data_set_training(self):
        """
        Load :class:`ImageDataSet` or :class:`ImageDataSetAugment for network input.
        Load :class:`DataLabel` as target.

        Returns:
            (tuple) -> (:class:`ImageDataSet` or :class:`ImageDataSetAugment`, :class:`DataLabel`)

        """
        img_out = self._read_image(self._input_dir)

        if not self.get_from_config('excel_sheetname', None) is None:
            gt_dat = MedImgDataset.DataLabel.from_xlsx(self._target_dir, self.get_from_config('excel_sheetname', ))
        else:
            gt_dat = MedImgDataset.DataLabel.from_csv(self._target_dir)

        # Load selected columns only
        if not self.get_from_loader_params('column') is None:
            self._logger.info("Selecting target column: {}".format(self.get_from_loader_params('column')))
            gt_dat.set_target_column(self.get_from_loader_params('column'))
        gt_dat.map_to_data(img_out)
        return img_out, gt_dat

