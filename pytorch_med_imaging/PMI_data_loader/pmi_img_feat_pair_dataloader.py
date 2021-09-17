from .pmi_image_dataloader import PMIImageDataLoader
from torch.utils.data import TensorDataset
from .. import med_img_dataset

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

    Loader_Params:
        excel_sheetname (Optional):
            Name of excel sheet if the target is an excel file
        column (Optional):
            A comma seperated string that specified the columns to read for ground-truth dataset
        net_in_label_dir (Optional):
            If this is specified, an extra set of data will be loaded and input to the network
        net_in_column (Optional):
            Same as `column` but used for the extra set of data.

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
        out = self._read_image(self._input_dir)

        if not self.get_from_loader_params('excel_sheetname', None) is None:
            gt_dat = med_img_dataset.DataLabel.from_xlsx(self._target_dir, self.get_from_config('excel_sheetname', ))
        else:
            gt_dat = med_img_dataset.DataLabel.from_csv(self._target_dir)

        # Load selected columns only
        if not self.get_from_loader_params('column') is None:
            self._logger.info("Selecting target column: {}".format(self.get_from_loader_params('column')))
            gt_dat.set_target_column(self.get_from_loader_params('column'))
        gt_dat.map_to_data(out)

        # Load extra column and concat if extra column options were found
        if not self.get_from_loader_params('net_in_label_dir') is None:
            self._logger.info("Selecting extra input columns")
            if not self.get_from_loader_params('net_in_excel_sheetname', None) is None:
                extra_dat = med_img_dataset.DataLabel.from_xlsx(self._target_dir, self.get_from_config('net_in_excel_sheetname', ))
            else:
                extra_dat = med_img_dataset.DataLabel.from_csv(self._target_dir)
            extra_dat.set_target_column(self.get_from_loader_params('net_in_column'))
            extra_dat.map_to_data(out)
            self._logger.info(f"extradat: {extra_dat.size()}")
            self._logger.info(f"out: {out}")
            return TensorDataset(out,extra_dat) ,gt_dat

        else:
            return out, gt_dat

    def _load_data_set_inference(self):
        # Load extra column and concat if extra column options were found
        if not self.get_from_loader_params('net_in_label_dir') is None:
            self._logger.info("Selecting extra input columns")
            if not self.get_from_loader_params('net_in_excel_sheetname', None) is None:
                extra_dat = med_img_dataset.DataLabel.from_xlsx(self._target_dir, self.get_from_config('net_in_excel_sheetname', ))
            else:
                extra_dat = med_img_dataset.DataLabel.from_csv(self._target_dir)
            extra_dat.set_target_column(self.get_from_loader_params('net_in_column'))
            im_dat = super(PMIImageFeaturePair, self)._load_data_set_inference()
            extra_dat.map_to_data(im_dat)

            return TensorDataset(im_dat, extra_dat)
        else:
            return super(PMIImageFeaturePair, self)._load_data_set_inference()

