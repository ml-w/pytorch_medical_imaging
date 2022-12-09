from .ImageData import ImageDataSet
from .PMIDataBase import PMIDataBase
import torch
from torch import cat, stack
import torchio as tio
import numpy as np
import os
from typing import *

class ImageDataMultiChannel(PMIDataBase):
    r"""
    ImageDataSetMultiChannel class allows you to load either vector images (Pending) or load from a bunch
    of subdirectories and concatanate them as multi-channel input. The channels will order accoding to the
    alphabetical order of the subdirecties. They will use the first set of data as references and load the ids that
    exist in the channel.

    You can also specify the subdirectories in form of a list, where all nii.gz files under
    those directories will be loaded according to the specified order.

    Args:
        rootdir (str):
            Path to the root directory for reading nifties
        channel_subdirs (list of str, Optional):
            A list of subdirectories under the rootdir for file loading. If `None`, load from the 1st layer
            of subdirectory. Default to `None`.
        readmode (str, Optional):
            Decide image directories globbing method, whether to look into subdirectories or not. \n
            Possible values:
            * `normal` - typical loading behavior, reading all nii/nii.gz files in the directory.
            * `recursive` - search all subdirectories excluding softlinks, use with causion.
            * `explicit` - specifying directories of the files to load.
            Default is `normal`.
        concat_by_axis (int, Optional)
            If value is >=0, the images will be concatenated at specified axis instead of the channel, 0 equals to
            th first axis after channel dimension.

    .. hint::
        * There are also other arguments required by :class:`ImageDataSet`, you must specify them correctly in order for
          this to work.
        * If you wish to use it with the dataloader, you should consider simply using :class:`PMIImageDataMCLoader`.

    Examples:

        1. Consider the following file structures::

            /rootdir
            └───Channel_sub_dir_1
            |  |   ID1_image.nii.gz
            |  |   ID2_image.nii.gz
            |  |   ...
            |
            └───Channel_sub_dir_2
            |  |   ID1_image.nii.gz
            |  |   ID2_image.nii.gz
            |  |   ...
            |
            └───Channel_sub_dir_3
            |   ID1_image.nii.gz
            |   ID2_image.nii.gz
            |   ...

        2. If ``channel_subdirs`` was not set, three :class:`ImageDataSet` objects will be created, each inherits all
           the remaining tags and options from the input arguments, with their `root_dir` set to the subdirs:

            >>> from pytorch_med_imaging.med_img_dataset import ImageDataMultiChannel
            >>> rootdir = 'rootdir'
            >>> imset = ImageDataSetMultiChannel(rootdir, channel_subdirs=['Channel_sub_dir_1', 'Channel_sub_dir_2'])

        3. Shape of the output from imset will be:

            >>> print(imset.shape)
            # (N, 2, D, W, H)

    See Also:
        * :class:`ImageDataSet`


    """

    def __init__(self,
                 rootdir,
                 channel_subdirs = None,
                 concat_by_axis = 0,
                 *args, **kwargs):
        super(ImageDataMultiChannel, self).__init__()

        # check input
        self._rootdir = rootdir
        if not os.path.isdir(self._rootdir):
            self._logger.error("Root dir specified {} not found.".format(self._rootdir))
            raise AssertionError("Root dir specified {} not found.".format(self._rootdir))

        self._channel_subdirs = channel_subdirs
        self._concat_by_axis = concat_by_axis
        if self._channel_subdirs is None:
            self._logger.info("Using all sub directories in specified root directory.")


        self._basedata = []
        if self._channel_subdirs is None:
            self._channel_subdirs = []
            for d in os.listdir(self._rootdir):
                # check if its a directory, if so, load as ImageDataSet
                if os.path.isdir(os.path.join(self._rootdir, d)):
                    self._logger.info("Loading from subdir: {}".format(d))
                    self._channel_subdirs.append(self._rootdir, d)
                else:
                    self._logger.info("Excluding non-directories in subdir: {}".format(d))
        else:
            assert all([os.path.isdir(os.path.join(self._rootdir, d)) for d in self._channel_subdirs]), \
                "Cannot open specified directory when loading subdirectories. {}".format(
                    {os.path.join(self._rootdir, d): os.path.isdir(os.path.join(self._rootdir, d)) for d in self._channel_subdirs}
                )


        # obtain first batch of ids
        self._logger.info("Testing first set of sub-dir data to define IDs.")
        self._basedata.append(ImageDataSet(os.path.join(self._rootdir,
                                                        self._channel_subdirs[0]),
                                           **kwargs))
        ids = self._basedata[0].get_unique_IDs()
        self._logger.debug("Extracted ids: {}".format(ids))

        # Use it to load remaining ids
        self._logger.info("Using extracted IDs for loading remaining data in {}.".format(self._channel_subdirs[1:]))
        for i in range(1, len(self._channel_subdirs)):
            _temp_dict = dict(kwargs)
            _filtermode = _temp_dict.get('filtermode', None)
            if _filtermode == None:
                _temp_dict['filtermode'] = 'idlist'
                _temp_dict['idlist'] = ids
            elif _filtermode == 'both':
                _temp_dict['idlist'] = ids
            elif _filtermode == 'idlist':
                _temp_dict['idlist'] = ids
            else:
                raise AttributeError("Incorrect specifications to filter.")

            self._basedata.append(ImageDataSet(os.path.join(self._rootdir,
                                                            self._channel_subdirs[i],
                                                            ),
                                               **_temp_dict))

        # Calculate size
        self._size = self._basedata[0].size()


        # Check if dtype are the same
        self._logger.info("Checking if data are all the same datatype...")
        self._UNIQUE_DTYPE = np.all([dat.type() == self._basedata[0].type() for dat in self._basedata])
        self._logger.info("{}".format(self._UNIQUE_DTYPE))
        if not self._UNIQUE_DTYPE:
            msg = f"Not all data are of the same datatype! {[self._basedata[0].type() for dat in self._basedata]}"
            raise TypeError(msg)

    def get_unique_IDs(self, globber: Optional[str] = None) -> Iterable[str]:
        r"""Get all IDs globbed by the specified globber. If its None,
        default globber used. If its not None, the class globber will be
        updated to the specified one.

        Args:
            globber (str):
                Regex pattern to glob ID from the loaded files. If `None`, the stored attribute
                :attribute:`_id_globber` will be used.


        Return:
            list: A sorted list of unique IDs globbed using `globber`.

        """
        return self._basedata[0].get_unique_IDs(globber)

    def size(self, i = None):
        r"""Required by pytorch dataloader."""
        return len(self._basedata[0])

    def Write(self, *args):
        try:
            self._basedata[0].write_all(*args)
        except Exception as e:
            self._logger.exception("Seems like basedata {} have no Write() method.".format(
                self._basedata[0].__class__.__name__))
            raise NotImplementedError("Base data have no Write() method")

    def __len__(self):
        return self.size()

    def __getitem__(self, item) -> torch.Tensor:
        return cat([dat[item] for dat in self._basedata], dim=self._concat_by_axis)

    def get_data_by_ID(self, *args):
        return cat([dat.get_data_by_ID(*args) for dat in self._basedata], dim=self._concat_by_axis)
