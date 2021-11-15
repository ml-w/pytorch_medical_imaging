from .pmi_image_dataloader import PMIImageDataLoader
from ..med_img_dataset import DataLabel
from torch.utils.data import TensorDataset
from .. import med_img_dataset

from typing import Optional
import torchio as tio

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


    def _read_params(self, config_file=None):
        # Image part is handled by parent class
        super(PMIImageFeaturePair, self)._read_params(config_file)

        default_attr = {
            'excel_sheetname': None,        # If the excel has multiple sheets
            'net_in_colname': None,         # Name(s) of the column(s) to input into the network
            'lossfunc_in_colname': None,    # Name(s) of the column(s) to input into the loss function
        }
        self._load_default_attr(default_attr)

    def _load_data_set_training(self,
                                exclude_augment: Optional[bool] = False) -> tio.Queue or tio.SubjectsDataset:
        """

        """
        if self._target_dir is None:
            raise IOError(f"Cannot load from {self._target_dir}")

        img_out = self._read_image(self._input_dir)
        mask_out = self._read_image(self._mask_dir, dtype='uint8')

        # Load the datasheet
        if not self.excel_sheetname is None:
            gt_dat = DataLabel.from_xlsx(self._target_dir, self.excel_sheetname)
        else:
            gt_dat = DataLabel.from_csv(self._target_dir)

        # Load selected columns only
        if not self.lossfunc_in_colname is None:
            self._logger.info("Selecting target column: {}".format(self.lossfunc_in_colname))
            gt_dat.set_target_column(self.lossfunc_in_colname)
        gt_dat.map_to_data(img_out)

        # Load extra column and concat if extra column options were found
        if not self.net_in_colname is None:
            self._logger.info(f"Selecting extra input columns: {self.net_in_colname}")
            if not self.excel_sheetname is None:
                extra_dat = DataLabel.from_xlsx(self._target_dir, self.excel_sheetname)
            else:
                extra_dat = DataLabel.from_csv(self._target_dir)
            extra_dat.set_target_column(self.net_incolname)
            extra_dat.map_to_data(img_out)
            self._logger.debug(f"extradat: {extra_dat.size()}")
            self._logger.debug(f"out: {img_out}")
        else:
            extra_dat = None

        self.data = {'input':   img_out,
                     'gt':      gt_dat,
                     'mask':    mask_out,
                     'net_in_dat': extra_dat,
                     'uid': img_out.get_unique_IDs()
                     }
        # create transform
        self._create_transform(exclude_augment=exclude_augment)

        # exclude where self.data items are `None`
        data_exclude_none = {k: v for k, v in self.data.items() if v is not None}

        # Create subject list
        subjects = [tio.Subject(**{k: v for k, v in zip(data_exclude_none.keys(), row)})
                    for row in zip(*data_exclude_none.values())]
        self._logger.debug(f"subjects: {subjects[0]}")
        subjects = tio.SubjectsDataset(subjects=subjects, transform=self.transform)

        return self._create_queue(exclude_augment, subjects)

    def _create_queue(self, exclude_augment, subjects):
        # Return the queue
        if not self.patch_size is None:
            overlap = [ps // 2 for ps in self.patch_size]
            # If no probmap, return GridSampler, otherwise, return weighted sampler
            if self.data['probmap'] is None:
                sampler = tio.GridSampler(patch_size=self.patch_size, patch_overlap=overlap)
            else:
                sampler = self.sampler
            return subjects, sampler
        else:
            # Set queue_args and queue_kwargs to load the whole image for each object to allow for caching
            shape_of_input = subjects[0].shape

            # Reset sampler
            self.sampler = tio.UniformSampler(patch_size=shape_of_input[1:])  # first dim is batch
            self.queue_args[-1] = self.sampler

            # if exclude augment, don't shuffle
            if exclude_augment:
                _inf_dic = self.queue_kwargs
                _inf_dic['shuffle_subjects'] = False
                _inf_dic['shuffle_subjects'] = False
            else:
                _inf_dic = self.queue_kwargs

            # Create queue
            queue = tio.Queue(subjects, *self.queue_args, **self.queue_kwargs)
            self._logger.debug(f"Created queue: {queue}")
            return queue

    def _load_data_set_inference(self) -> tio.Queue or tio.SubjectsDataset:
        img_out = self._read_image(self._input_dir)
        mask_out = self._read_image(self._mask_dir, dtype='uint8')

        # TODO: net_in_dat was assume to be in the same excel file as target_dir, which is not correct assumption
        self.data = {'input':   img_out,
                     'mask':    mask_out,
                     'uid': img_out.get_unique_IDs()
                     }

        # create transform
        self._create_transform(exclude_augment=True)

        # Create subject list
        data_exclude_none = {k: v for k, v in self.data.items() if v is not None}
        subjects = [tio.Subject(**{k: v for k, v in zip(data_exclude_none.keys(), row)})
                    for row in zip(*data_exclude_none.values())]
        subjects = tio.SubjectsDataset(subjects=subjects, transform=self.transform)

        return self._create_queue(exclude_augment, subjects)
