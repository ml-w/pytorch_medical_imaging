from .ImageData import ImageDataSet
from .PMIDataBase import PMIDataBase
from torch import cat, stack
import numpy as np
import os


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
        filtermode (str, Optional):
            After grabbing file directories, they are filtered by either ID, regex or both. Corresponding att needed. \n
            Usage:
                * `idlist`: Extract images that is on a specified list, globbed with `idGlobber`. Requires att `idlist`.
                * `regex`: Extract images that matches one regex sepcified with att `regex`.
                * `both': Use both `idlist` and `regex` as filtering method. Requires both att specified.
                * None: No filter, read all .nii.gz images in the directory.
            Default is `None`.
        idlist (str or list, Optional):
            If its `str`, it should be directory to a file containing IDs, one in each line, otherwise,
            an explicit list of strings. Need if filtermode is 'idlist'. Globber of id can be specified with attribute
            idGlobber.
        regex (str, Optional):
            Regex that is used to match file directories. Un-matched ones are discarded. Effective when
            `filtermode='idlist'`.Must start with paranthesis. Otherwise, its treated as wild cards, e.g. `'*nii.gz'`
        idGlobber (str, Optional):
            Regex string to search ID. Effective when filtermode='idlist', optional. If none specified
            the default globber is `'(^[a-ZA-Z0-9]+)`, globbing the first one matches the regex in file basename. .
        loadBySlices (int, Optional):
            If its < 0, images are loaded as 3D volumes. If its >= 0, the slices along i-th dimension loaded. Default is `-1`
        verbose (bool, Optional):
            Whether to report loading progress or not. Default to `False`.
        dtype (str or type, Optional):
            Cast loaded data element to the specified type. Default is `float`.
        debugmode (bool, Optional):
            For debug only. Default is `False`
        recursiveSearch (bool, Optional):
            Whether to load files recursively into subdirectories. Default is `False`


    Examples:

        1. Consider the following file structures:
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

        2. If `channel_subdirs` was not set, three :class:`ImageDataSet` objects will be created, each inherits all the
            remaining tags and options from the input arguments, with their `root_dir` set to the subdirs.

            >>> from pytorch_med_imaging.med_img_dataset import ImageDataMultiChannel
            >>> rootdir = 'rootdir'
            >>> imset = ImageDataSetMultiChannel(rootdir)

        3. Shape of the output from imset will be:
            >>> print(imset.shape)
            # (N, 3, D, W, H)

    See Also:
        :class:`ImageDataSet`


    """

    def __init__(self, *args, **kwargs):
        super(ImageDataMultiChannel, self).__init__()

        # check input
        self._rootdir = args[0]
        if not os.path.isdir(self._rootdir):
            self._logger.error("Root dir specified {} not found.".format(self._rootdir))
            raise AssertionError("Root dir specified {} not found.".format(self._rootdir))

        self._channel_subdirs = None
        if 'channel_subdirs' in kwargs:
            self._logger.info("Using subdir options.")
            self._channel_subdirs = kwargs['channel_subdirs']
        else:
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
        self._itemindexes = self._basedata[0]._itemindexes
        ids = self._basedata[0].get_unique_IDs()
        self._logger.debug("Extracted ids: {}".format(ids))

        # Use it to load remaining ids
        self._logger.info("Using extracted IDs for loading remaining data in {}.".format(self._channel_subdirs[1:]))
        for i in range(1, len(self._channel_subdirs)):
            _temp_dict = dict(kwargs)
            if _temp_dict['filtermode'] == None:
                _temp_dict['filtermode'] = 'idlist'
                _temp_dict['idlist'] = ids
            elif _temp_dict['filtermode'] == 'both':
                _temp_dict['idlist'] = ids
            elif _temp_dict['filtermode'] == 'idlist':
                _temp_dict['idlist'] = ids
            else:
                raise AttributeError("Incorrect specifications to filter.")

            self._basedata.append(ImageDataSet(os.path.join(self._rootdir,
                                                            self._channel_subdirs[i],
                                                            ),
                                               **_temp_dict))

        # Options
        self._concat_by_axis = kwargs.get('concat_by_axis', -1)   # concated at the specified axis if > -1
        if self._concat_by_axis > -1:
            self._logger.info(f"Concatenating output at axis {self._concat_by_axis} instead of channels.")

        # Inherit some of the properties of the inputs
        self._byslices = kwargs['loadBySlices']

        # Calculate size
        self._size = list(self._basedata[0].size())
        self._size[1] = len(self._basedata)

        # Check if dtype are the same
        self._logger.info("Checking if data are all the same datatype...")
        self._UNIQUE_DTYPE = np.all([dat.type() == self._basedata[0].type() for dat in self._basedata])
        self._logger.info("{}".format(self._UNIQUE_DTYPE))

    def get_unique_IDs(self, globber=None):
        return self._basedata[0].get_unique_IDs(globber)

    def size(self, int=slice(None)):
        return self._size[int]

    def Write(self, *args):
        try:
            self._basedata[0].write_all(*args)
        except Exception as e:
            self._logger.exception("Seems like basedata {} have no Write() method.".format(
                self._basedata[0].__class__.__name__))
            raise NotImplementedError("Base data have no Write() method")

    def __len__(self):
        return self.size()[0]

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = [item.start ,item.stop, item.step]
            start = start if not start is None else 0
            stop = stop if not stop is None else len(self._basedataset)
            step = step if not step is None else 1

            return stack([self.__getitem__(i) for i in range(start, stop, step)])
        else:
            # output datatype will follow the dtype of the first constructor argument
            if self._UNIQUE_DTYPE:
                if self._concat_by_axis >= 0:
                    return cat([dat[item] for dat in self._basedata], dim=self._concat_by_axis + 2)
                else:
                    return cat([dat[item] for dat in self._basedata])
            else:
                self._logger.debug("Performing type-cast for item {}.".format(item))
                if self._concat_by_axis >= 0:
                    cat([dat[item].type_as(self._basedata[0][item]) for dat in self._basedata],
                        dim=self._concat_by_axis)
                else:
                    return cat([dat[item].type_as(self._basedata[0][item]) for dat in self._basedata])