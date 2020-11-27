from .PMIDataLoaderBase import PMIDataLoaderBase
from .. import med_img_dataset
from ..med_img_dataset.computations import ImageDataSetWithTexture

import re

__all__ = ['PMIImageDataLoader']

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
        self._exclude = self.get_from_config('Filters', 'id_exclude', None)
        if not self._exclude is None:
            self._exclude = self._exclude.split(',')
            for e in self._exclude:
                if e in self._idlist:
                    self._logger.info("Removing {} from the list as specified.".format(e))
                    self._idlist.remove(e)



        self._data_subtype = self.get_from_loader_params('data_subtype', None)
        self._augmentation = self.get_from_loader_params_with_eval('augmentation', 0)
        self._load_by_slices = self.get_from_loader_params_with_eval('load_by_slices', -1)
        self._load_with_filter = self.get_from_loader_params('load_with_filter', "")
        self._results_only = self.get_from_loader_params_with_boolean('results_only', False)
        self._channel_first = self.get_from_loader_params_with_boolean('channel_first', False)

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
        if self._augmentation > 0 and not re.match('(?=.*train.*)', self._run_mode) is None:
            self._image_class = med_img_dataset.ImageDataSetAugment
        else:
            self._image_class = med_img_dataset.ImageDataSet

        img_data =  self._image_class(root_dir, verbose=self._verbose, debugmode=self._debug, filtermode='both',
                                 regex=self._regex, idlist=self._idlist, loadBySlices=self._load_by_slices,
                                 aug_factor=self._augmentation, **kwargs)

        if re.search("texture", self._load_with_filter, flags=re.IGNORECASE) is not None:
            img_data = ImageDataSetWithTexture(img_data, results_only=self._results_only, channel_first=self._channel_first)
        return img_data

    def _load_data_set_training(self):
        """
        Load either ImageDataSet or ImageDataSetAugment for network input.

        Detect if loading type is a segmentation, if so, cast ground-truth data to `uint8` and mark them as
        segmentation such that the augmentator adjust for it.

        Returns:
            (ImageDataSet or ImageDataSetAugment)

        """
        if self._target_dir is None:
            raise AttributeError("Object failed to load _target_dir.")

        img_out = self._read_image(self._input_dir)
        if not re.match("(?=.*seg.*)", self._data_subtype, re.IGNORECASE) is None:
            gt_out = self._read_image(self._target_dir, dtype='uint8', is_seg=True, reference_dataset=img_out)
        elif not re.match("(?=.*encoder.*)", self._data_subtype, re.IGNORECASE) is None:
            gt_out = img_out
        else:
            gt_out = self._read_image(self._target_dir, reference_dataset=img_out)

        return img_out, gt_out


    def _load_data_set_inference(self):
        """Same as :func:`_load_data_set_training` in this class."""
        return self._read_image(self._input_dir)

