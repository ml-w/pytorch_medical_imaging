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
        _regex (str): Please see :class:`ImageDataSet`.
        _idlist (str or list): Please see :class:`ImageDataSet`.
        _augmentation (int): If `_augmentation` > 0, :class:`ImageDataSetAugment` will be used instead.
        _load_by_slice (int):
            If `_load_by_slice` > -1, images volumes are loaded slice by slice along the axis specified.

    .. note::
        These attributes are defined in :func:`PMIImageDataLoader._read_params`


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
        see the super class for more details.

        Args:
            config_file (str or dict, Optional): See :func:`PMIDataLoaderBase._read_params`.

        See Also:
            * :class:`PMIDataLoaderBase`
            * :func:`PMIDataLoaderBase._read_params`
        """

        super(PMIImageDataLoader, self)._read_params(config_file)

        self._regex = self.get_from_prop_dict('regex', None)
        self._idlist = self.get_from_prop_dict('idlist', None)
        if isinstance(self._idlist, str):
            self.idlist = self.parse_ini_filelist(self._idlist, self._run_mode)

        self._augmentation = self.get_from_prop_dict('augmentation', 0)
        self._load_by_slices = self.get_from_prop_dict('load_by_slices', -1)

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
        if self._augmentation > 0:
            self._image_class = ImageDataSetAugment
        else:
            self._image_class = ImageDataSet

        return self._image_class(root_dir, verbose=self._verbose, debug=self._debug, filtermode='both',
                                 regex=self._regex, idlist=self._idlist, loadBySlices=self._load_by_slices,
                                 aug_factor=self._augmentation, **kwargs)

    def _load_data_set_training(self):
        return self._read_image(self._input_dir)

    def _load_data_set_loss_func_gt(self):
        if not re.match("(?=.*seg.*)", self._datatype, re.IGNORECASE) is None:
            return self._read_image(self._target_dir, dtype='uint8', is_seg=True)
        else:
            return self._read_image(self._target_dir)

    @staticmethod
    def parse_ini_filelist(filelist, mode):
        r"""
        Parse the ini file for this class.

        Args:
            filelist (str): Relative directory to the ini filelist.

        Returns:
            (list): A list containing the IDs specifed in the target file list.

        Examples:

            An example of .ini file list should look something like this,

            file_list.ini::

                [FileList]
                testing=ID_0,ID_1,...,ID_n
                training=ID_a,ID_b,...,ID_m

        """
        assert os.path.isfile(filelist)

        fparser = configparser.ConfigParser()
        fparser.read(filelist)

        # test
        if re.match('(?=.*train.*)', mode):
            return fparser['FileList'].get('testing').split(',')
        else:
            return fparser['FileList'].get('training').split(',')
