from .pmi_image_dataloader import PMIImageDataLoader, PMIImageDataLoaderCFG
from ..med_img_dataset import DataLabel, DataLabelConcat

from typing import Optional
import torchio as tio
from pathlib import Path

__all__ = ['PMIImageFeaturePairLoader', 'PMIImageFeaturePairLoaderCFG', 'PMIImageFeaturePairLoaderConcat']


class PMIImageFeaturePairLoaderCFG(PMIImageDataLoaderCFG):
    r"""Configuration for :class:`PMIImageFeaturePairLoader`.

    Class Attributes:
        excel_sheetname (str, Optional):
            The name of the target excel sheet that is specified in ``target_dir``. See also :class:`PMIDataLoaderBase`.
        target_column (str, Optional):
            A comma seperated string that specified the columns to read for ground-truth dataset. If not specified, use
            all columns in the first sheet of the excel file.
        net_in_column (str, Optional):
            Specify the column in the excel sheet that needs to go into the network. Default to ``None``.
        net_in_label_dir (str, Optional):
            If this is specified, an extra set of data will be loaded and input to the network. Defaul to ``None``.
        net_in_dtype (type, Optional):
            Force type cast for net_in_column.
    """
    excel_sheetname   : str = None
    target_column     : str = None
    net_in_column     : str = None
    net_in_label_dir  : str = None
    net_in_dtype      : type = None

class PMIImageFeaturePairLoader(PMIImageDataLoader):
    """
    This class load :class:`ImageDataSet` related image data together with features written in a csv file specified
    by ``cfg.target_dir``.

    This class is suitable in the following situations:
        * Image to features prediction
        * Image classifications
        * Image to coordinates

    For attributes and configurations, see :class:`PMIImageFeaturePairLoaderCFG`

    Args:
        cfg (PMIImageFeaturePairLoaderCFG): Config file instant/class. See :class:`PMIImageFeaturePairLoaderCFG`.
        *args: Please see parent class.
        **kwargs: Please see parent class.

    See Also:
        :class:`PMIDataLoaderBase`

    """
    def __init__(self, cfg: PMIImageFeaturePairLoaderCFG, *args, **kwargs):
        super(PMIImageDataLoader, self).__init__(cfg, *args, **kwargs)

    def _prepare_data(self) -> dict:
        r"""Override to change behavior of data loaders. See also :func:`PMIImageFeaturePairLoader._load_gt_data`.

        Returns:
            dict
        """
        data = super(PMIImageFeaturePairLoader, self)._prepare_data()

        # Load selected columns only
        gt_dat = data['gt']
        if not self.target_column in (None, ""):
            self._logger.info("Selecting target column: {}".format(self.target_column))
            try:
                dtype = self.data_types[1]
            except IndexError:
                dtype = None
            gt_dat.set_target_column(self.target_column, dtype=dtype)
        gt_dat.map_to_data(data['input'])

        # Load extra column and concat if extra column options were found
        if not self.net_in_column is None:
            self._logger.info(f"Selecting extra input columns: {self.net_in_column}")
            if not self.excel_sheetname is None:
                extra_dat = DataLabel.from_xlsx(self.target_dir, self.excel_sheetname)
            else:
                extra_dat = DataLabel.from_csv(self.target_dir)
            extra_dat.set_target_column(self.net_in_colname, dtype=self.net_in_dtype)
            extra_dat.map_to_data(img_out)
            self._logger.debug(f"extradat: {extra_dat.size()}")
            self._logger.debug(f"out: {img_out}")
        else:
            extra_dat = None

        # Add extra objects to self.data
        data['net_in_dat'] = extra_dat
        return data

    def _load_gt_data(self):
        # Load the datasheet
        if Path(self.target_dir).suffix == '.csv':
            gt_dat = DataLabel.from_csv(self.target_dir)
        elif not self.excel_sheetname is None:
            gt_dat = DataLabel.from_xlsx(self.target_dir, self.excel_sheetname)
        return gt_dat


class PMIImageFeaturePairLoaderConcat(PMIImageFeaturePairLoader):
    r"""Basically same as the base class but change from using `DataLabel` to `DataLabelConcat`, which is for the
    circumstance where the one data point span across multiple rows of the target column.

    This class is suitable for:
    * img to sequence
    """
    def __init__(self, *args, **kwargs):
        super(PMIImageFeaturePairLoaderConcat, self).__init__(*args, **kwargs)

    def _read_config(self, config_file=None):
        super(PMIImageFeaturePairLoaderConcat, self)._read_config(config_file)

    def _load_gt_data(self):
        # Load the datasheet
        if Path(self.target_dir).suffix == '.xlsx':
            gt_dat = DataLabelConcat.from_xlsx(self.target_dir, sheet_name=self.excel_sheetname)
        elif Path(self.target_dir).suffix == '.csv':
            gt_dat = DataLabelConcat.from_csv(self.target_dir)
        return gt_dat