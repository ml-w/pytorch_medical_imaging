from .PMIDataLoaderBase import PMIDataLoaderBase
from MedImgDataset import ImageDataSet, ImageDataSetAugment

import re
import os
import configparser

class PMIImageDataLoader(PMIDataLoaderBase):
    """
    This class load :class:ImageDataSet related image data. Customization of loading this class should inherit this
    class.

    This class is suitable in the following situations:
        * Image to image
        * Image to segmentation
        * Image super-resolution

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

    def _check_input(self):
        """Not implemented."""
        return True

    def _read_params(self, config_file=None):
        """
        Defines attributes. Called when object is created. Extra attributes are declared in super function,
        see the super class for more details. Params are read from `[LoaderParams]` section of the ini.

        Args:
            config_file (str or dict, Optional): See :func:`PMIDataLoaderBase._read_params`.

        See Also:
            * :class:`PMIDataLoaderBase`
            * :func:`PMIDataLoaderBase._read_params`
        """

        super(PMIImageDataLoader, self)._read_params(config_file)
        self._regex = self.get_from_config('Filters', 're_suffix', None)
        self._idlist = self.get_from_config('Filters', 'id_list', None)
        if isinstance(self._idlist, str):
            if self._idlist.endswith('.ini'):
                self._idlist = self.parse_ini_filelist(self._idlist, self._run_mode)
            elif self._idlist.endswith('.txt'):
                self._idlist = [r.rstrip() for r in open(self._idlist).readlines()]
            else:
                self._idlist = self._idlist.split(',')
            self._idlist.sort()

        self._augmentation = self.get_from_loader_params_with_eval('augmentation', 0)
        self._load_by_slices = self.get_from_loader_params_with_eval('load_by_slices', -1)

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
        if self._augmentation > 0 and not re.match('(?=.*train.*)', self._run_mode) is None:
            self._image_class = ImageDataSetAugment
        else:
            self._image_class = ImageDataSet

        return self._image_class(root_dir, verbose=self._verbose, debugmode=self._debug, filtermode='both',
                                 regex=self._regex, idlist=self._idlist, loadBySlices=self._load_by_slices,
                                 aug_factor=self._augmentation, **kwargs)

    def _load_data_set_training(self):
        """
        Load either ImageDataSet or ImageDataSetAugment for network input.

        Detect if loading type is a segmentation, if so, cast ground-truth data to `uint8` and mark them as
        segmentation such that the augmentator adjust for it.

        Returns:
            (ImageDataSet or ImageDataSetAugment)

        """
        img_out = self._read_image(self._input_dir)
        if not re.match("(?=.*seg.*)", self._datatype, re.IGNORECASE) is None:
            gt_out = self._read_image(self._target_dir, dtype='uint8', is_seg=True, reference_dataset=img_out)
        else:
            gt_out = self._read_image(self._target_dir, reference_dataset=img_out)

        return img_out, gt_out


    def _load_data_set_inference(self):
        """Same as :func:`_load_data_set_training` in this class."""
        return self._read_image(self._input_dir)

