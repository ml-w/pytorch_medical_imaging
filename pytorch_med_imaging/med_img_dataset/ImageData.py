from typing import Any, Iterable, Optional, Union

import torch
from torch import cat, unique

from .PMIDataBase import PMIDataBase

import pandas as pd
import torchio as tio
from tqdm.auto import tqdm
import fnmatch, re
import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib

NIFTI_DICT = {
    "sizeof_hdr": int,
    "data_type": str,
    "db_name": str,
    "extents": int,
    "session_error": int,
    "regular": str,
    "dim_info": str,
    "dim": int,
    "intent_p1": float,
    "intent_p2": float,
    "intent_p3": float,
    "intent_code": int,
    "datatype": int,
    "bitpix": int,
    "slice_start": int,
    "pixdim": float,
    "vox_offset": float,
    "scl_slope": float,
    "scl_inter": float,
    "slice_end": int,
    "slice_code": str,
    "xyzt_units": str,
    "cal_max": float,
    "cal_min": float,
    "slice_duration": float,
    "toffset": float,
    "glmax": int,
    "glmin": int,
    "descrip": str,
    "aux_file": str,
    "qform_code": int,
    "sform_code": int,
    "quatern_b": float,
    "quatern_c": float,
    "quatern_d": float,
    "qoffset_x": float,
    "qoffset_y": float,
    "qoffset_z": float,
    "srow_x": str,
    "srow_y": str,
    "srow_z": str,
    "intent_name": str,
    "magic": str
}

class ImageDataSet(PMIDataBase):
    r"""
    ImageDataSet class that reads and load nifty in a specified directory.

    Attributes:
        root_dir (str):
            Root dir of image loading.
        data_source_path (list of str):
            Directories of the input image relative to the root dir
        data (torch.tensor or list of torch.tensor):
            Actual data to load from. It will be one stack of images if image dimensions are compatible or if
            specified to load data slices by slices.
        metadata (list of dict):
            Meta data of the loaded images. Such as their origin, orientation and dimensions...etc.

    Args:
        rootdir (str):
            Path to the root directory for reading nifties
        readmode (str, Optional):
            Decide image directories globbing method, whether to look into subdirectories or not.
            Possible values:
                * `normal` - [Default] typical loading behavior, reading all nii/nii.gz files in the directory.
                * `recursive` - search all subdirectories excluding softlinks, use with causion.
                * `explicit` - specifying directories of the files to load.
        filtermode (str, Optional):
            After grabbing file directories, they are filtered by either ID, regex or both. Corresponding att needed.
            Usage:
                * `idlist`: Extract images that is on a specified list, globbed with `id_globber`. Requires att `idlist`.
                * `regex`: Extract images that matches one regex sepcified with att `regex`.
                * `both': Use both `idlist` and `regex` as filtering method. Requires both att specified.
                * `None`: [Default] No filter, read all .nii.gz images in the directory.
        idlist (str or list, Optional):
            If its `str`, it should be directory to a file containing IDs, one in each line, otherwise,
            an explicit list of strings. Need if filtermode is 'idlist'. Globber of id can be specified with attribute
            id_globber.
        regex (str, Optional):
            Regex that is used to match file directories. Un-matched ones are discarded. Effective when
            `filtermode='idlist'`.Must start with paranthesis. Otherwise, its treated as wild cards, e.g. `'*nii.gz'`
        id_globber (str, Optional):
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
    ---------

        **Loading images:**

        1. Load all nii images in a folder:

            >>> from pytorch_med_imaging.med_img_dataset import ImageDataSet
            >>> imgset = ImageDataSet('/some/dir/')

        2. Load all nii images, filtered by string 'T2W' in string:

            >>> imgset = ImageDataSet('/some/dir/', filtermode='regex', regex='(?=.*T2W.*)')

        3. Given a text file '/home/usr/loadlist.txt' with all image directories, say 'load.txt', load all nii
           images, filtered by file basename having numbers::

            # load.txt
            /home/usr/img1.nii.gz
            /home/usr/img2.nii.gz
            /home/usr/temp/img3.nii.gz
            /home/usr/temp/img_not_wanted.nii.gz

            # Commend to load.
            >>> imgset = ImageDataSet('/home/usr/loadlist.txt', readmode='explicit')


        **Geting image from object:**

        1. Getting the first and second images:

            >>> im_1st = imgset[0]
            >>> im_2nd = imgset[1]
            >>> type(im_1st)
            torch.tensor

        2. Getting the last image:

            >>> imset[-1]
            torch.tensor

        3. Print details of the loaded images:

            >>> print(imset)

    .. hint::
        Use ``instance[item]`` to get the data as ``torch.Tensor``.
    """
    def __init__(self, rootdir, readmode='normal', filtermode=None, verbose=False, dtype=float,
                 debugmode=False, **kwargs):
        super(ImageDataSet, self).__init__(verbose=verbose)
        assert os.path.isdir(rootdir), "Cannot access directory: {}".format(rootdir)

        self.rootdir            = rootdir
        self.data_source_path   = []
        self.data               = []
        self.metadata           = []
        self.metadata_table     = None
        self.length             = 0
        self.verbose            = verbose
        self.dtype              = dtype
        self._raw_length        = 0 # length of raw input (i.e. num of nii.gz files loaded)
        self._filterargs        = kwargs
        self._filtermode        = filtermode
        self._readmode          = readmode
        self._id_globber        = kwargs.get('id_globber', "(^[a-zA-Z0-9]+)")
        self._debug             = debugmode

        self._error_check()
        self._parse_root_dir()

    def _error_check(self):
        assert self._readmode in ['normal', 'recursive', 'explicit'], 'Wrong readmode specified.'

        if self._readmode == 'normal' or self._readmode == 'recursive':
            assert os.path.isdir(self.rootdir)

        if self._readmode == 'explicit':
            assert os.path.isfile(self.rootdir)

        assert self._filtermode in ['idlist', 'regex', 'both', None], 'Wrong filtermode specified.'
        assert self._filtermode in self._filterargs or self._filtermode in ['both', None], \
            'Specifying arguemnets to filter is necessary'
        if self._filtermode == 'both':
            assert all([ k in self._filterargs for k in ['idlist', 'regex']]), 'No filter Args.'


    def _parse_root_dir(self):
        r"""
        Main parsing function.
        """

        self._logger.info("Parsing root path: " + self.rootdir)

        #===================================
        # Read all nii.gz files exist first.
        #-----------------------------------
        removed_fnames = []
        if self._readmode == 'normal':
            file_dirs = os.listdir(self.rootdir)
            file_dirs = fnmatch.filter(file_dirs, "*.nii.gz")
            file_dirs = [os.path.join(self.rootdir, f) for f in file_dirs]
        elif self._readmode == 'explicit':
            file_dirs = [fs.rstrip() for fs in open(self.rootdir, 'r').readlines()]
            for fs in file_dirs:
                if not os.path.isfile(fs):
                    file_dirs.remove(fs)
                    removed_fnames.append(fs)
        elif self._readmode == 'recursive':
            file_dirs = []
            for root, folder, files in os.walk(self.rootdir):
                if len(files):
                    file_dirs.extend([os.path.join(root,f) for f in files])
            file_dirs = fnmatch.filter(file_dirs, '*.nii.gz')
        else:
            raise AttributeError("file_dirs is not assigned!")

        if len(file_dirs) == 0:
            self._logger.error("No target files found in {}.".format(self.rootdir))
            raise ArithmeticError("No target files found in {}.".format(self.rootdir))

        #==========================
        # Apply filter if specified
        #--------------------------
        filtered_away = []
        file_dirs = self._filter_filelist(file_dirs, filtered_away, removed_fnames)

        self._logger.info("Found %s nii.gz files..."%len(file_dirs))
        self._logger.info("Start Loading")


        #=============
        # Reading data
        #-------------
        self._itemindexes = [0] # [image index of start slice]
        for i, f in enumerate(tqdm(file_dirs, disable=not self.verbose, desc="Load Images")) \
                if not self._debug else enumerate(tqdm(file_dirs[:10],
                                                       disable=not self.verbose,
                                                       desc="Load Images")):
            if self.verbose:
                self._logger.info("Reading from "+f)

            if not os.path.isfile(f):
                self._logger.warning("Cannot find file!")
                self._logger.debug(f"{os.listdir(os.path.dirname(f))}")

            # if dtype is uint, treat as label
            if np.issubdtype(self.dtype, np.unsignedinteger):
                im = tio.LabelMap(f)
            else:
                im = tio.ScalarImage(f, check_nans=True)
            self.data_source_path.append(f)
            self.data.append(im)

            # read metadata
            nib_im = nib.load(f)
            im_header = nib_im.header
            im_header_dict = {key: im_header.structarr[key].tolist() for key in im_header.structarr.dtype.names}
            im_header_dict['orientation'] = im.orientation
            self.metadata.append(im_header_dict)

            self._raw_length += 1
        self.length = len(self.data_source_path)
        self._logger.info("Finished loading. Loaded {} files.".format(self.length))
        self._logger.debug(f"IDs of loaded images: {','.join(self.get_unique_IDs())}")

    def _filter_filelist(self, file_dirs, filtered_away, removed_fnames):
        r"""Filter the `file_dirs` using the specified attributions. Used in `parse_root_dir`."""
        # Filter by filelist
        #-------------------
        if (self._filtermode == 'idlist' or self._filtermode == 'both') and \
                self._filterargs['idlist'] not in ("", None) and self._id_globber is not None:
            self._logger.info("Globbing ID with globber: " + str(self._id_globber) + " ...")
            file_basenames = [os.path.basename(f) for f in file_dirs]
            file_ids = {f: re.search(self._id_globber, f) for f in file_basenames}
            file_ids = {f: v.group() for f, v in file_ids.items() if v is not None}
            if len(file_ids) != len(file_basenames):
                self._logger.warning("Not all files were assigned an ID.")
                no_ids = set(file_basenames) - set(list(file_ids.keys()))
                self._logger.debug(f"{no_ids}")
                self._logger.debug(f"{file_ids}")
            # Don't sort otherwise the order of file_ids and file_dirs will become different.
            file_ids = list(file_ids.values())

            if isinstance(self._filterargs['idlist'], str) and not self._filterargs['idlist'] == "":
                # If its a file directory
                self._idlist = [r.strip() for r in open(self._filterargs['idlist'], 'r').readlines()]
            elif isinstance(self._filterargs['idlist'], (list, tuple)):
                # If its a list of IDs
                self._idlist = self._filterargs['idlist']
            elif self._filterargs['idlist'] in (None, ""):
                # If None specified, glob ids from filenames instead
                self._logger.warning('Idlist input is None!')
                self._idlist = file_ids
            else:
                raise TypeError(f"ID list is not correclty spefified. Expect str, list or None, got "
                                f"{self._filterargs['idlist']} instead")

            self._logger.debug(f'Target IDs: {self._idlist}')
            self._logger.debug(f'All IDs: {file_ids}')
            self._logger.debug(f"Missing ID(s): {set(self._idlist) - set(file_ids)}")
            tmp_file_dirs = np.array(file_dirs)
            keep = [id in self._idlist for id in file_ids]  # error near this could be because nothing is grabed

            if len(file_dirs) != len(keep):
                raise IndexError("Number of files is different from number of globbed IDs!")
            file_dirs = tmp_file_dirs[keep].tolist()
            filtered_away.extend(tmp_file_dirs[np.invert(keep)])
            self._logger.debug(f"Filtering away: {filtered_away}")

            # Check if there are still things in the list
            if len(file_dirs) == 0:
                self._logger.warning("Nothing lefted in the file list after id-filtering! "
                                     "That can't be right, continue with all files found.")
                file_dirs = tmp_file_dirs.tolist()

        # Fitlter by regex
        # --------------
        if self._filtermode == 'regex' or self._filtermode == 'both':
            self._logger.info("Filtering ID with filter: {}".format(self._filterargs['regex']))
            file_basenames = [os.path.basename(f) for f in file_dirs]
            # use REGEX if find paranthesis
            if self._filterargs['regex'] is None:
                # do nothing if regex is Nonw
                self._logger.warning('Regex input is None!')
                pass
            # if find *, treat it as wild card, if find .* treat it as regex
            elif self._filterargs['regex'].find('*') == -1 or self._filterargs['regex'].find('.*') > -1:
                try:
                    keep = np.invert([re.match(self._filterargs['regex'], f) is None for f in file_basenames])
                except Exception as e:
                    import sys, traceback as tr
                    cl, exc, tb = sys.exc_info()
                    self._logger.error(f"Error encountered when performing regex filtering.")
                    self._logger.debug(f"Regex was {self._filterargs['regex']}")
                    self._logger.debug(f"Filenames were {file_basenames}")
                    self._logger.exception()

                try:
                    filtered_away.extend(np.array(file_dirs)[np.invert(keep)].tolist())
                except IndexError:
                    self._logger.exception("Error when trying to filter by regex.")
                    self._logger.debug(f"keep: {keep}")
                    self._logger.debug(f"file_dirs: {file_dirs}")
                except:
                    self._logger.exception("Unknown error when trying to filter by regex.")
                tmp_file_dirs = np.array(file_dirs)
                file_dirs = tmp_file_dirs[keep].tolist()
                filtered_away.extend(tmp_file_dirs[np.invert(keep)])
            else:  # else use wild card
                tmp_file_dirs = np.array(file_dirs)
                file_dirs = fnmatch.filter(file_dirs, "*" + self._filterargs['regex'] + "*")
                filtered_away.extend(list(set(tmp_file_dirs) - set(file_dirs)))

            # Check if there are still things in the list
            if len(file_dirs) == 0:
                self._logger.warning(
                    "Nothing lefted in the file list after regex-filtering! "
                    "That can't be right, continue with all files found.")
                file_dirs = tmp_file_dirs.tolist()

        if len(removed_fnames) > 0:
            removed_fnames.sort()
            for fs in removed_fnames:
                self._logger.warning("Cannot find " + fs + " in " + self.rootdir)
        file_dirs.sort()
        self._logger.debug(f"Reading from: {file_dirs}")
        return file_dirs

    def get_raw_data_shape(self) -> list:
        r"""Get shape of all files as a list (ignore load by slice option).

        Returns:
            list[tuples]
        """
        return [self.get_size(i) for i in range(len(self.metadata))]

    def check_shape_identical(self, target_imset: Any) -> bool:
        r"""Check if file shape is identical to another ImageDataSet. Convinient for checking if
        the ground-truth and the inputs are of the same size.

        .. TODO:
            * Add spacing, origin check into this function (Hint: see match_dimension.py)

        Args:
            target_imset (ImageDataSet):
                Target image dataset to compare with.

        Returns:
            bool
        """
        assert isinstance(target_imset, ImageDataSet), "Target is not image dataset."

        self_shape = self.get_raw_data_shape()
        target_shape = target_imset.get_raw_data_shape()

        if len(self_shape) != len(target_shape):
            self._logger.warning("Difference data length!")
            return False

        assert type(self_shape) == type(target_shape), "There are major discrepancies in dimension!"
        truth_list = [a == b for a, b in zip(self_shape, target_shape)]
        if not all(truth_list):
            discrip = np.argwhere(np.array(truth_list) == False)
            for x in discrip:
                self._logger.warning(
                            "Discripency in element %i, ID: %s, File:[%s, %s]" % (
                                x,
                                self.get_unique_IDs(x),
                                os.path.basename(self.get_data_source(x)),
                                os.path.basename(target_imset.get_data_source(x))))

        return all(truth_list)

    def size(self, i=None) -> Union[int, torch.Size]:
        r"""Required by pytorch dataloader.

        Returns:
            Union[int, torch.Size]
        """
        if i is None:
            try:
                return self.data.shape
            except:
                return self.length
        else:
            return self.length

    def type(self) -> Any:
        r"""Return datatype of the elements."""
        return self.data[0].type()

    def as_type(self, t) -> None:
        r"""Cast all elements to specified type."""
        try:
            self.data = self.data.type(t)
            self.dtype = t
        except Exception as e:
            self._logger.error("Error encounted during type cast.")
            self._logger.log_traceback()

    def get_data_source(self, i) -> str:
        r"""Get directory of the source of the i-th element, sorted by filenames.

        Args:
            i (int): Index.

        Returns:
            str

        """
        return self.data_source_path[i]

    def get_data_by_ID(self,
                       id: str,
                       globber: Optional[str] = None,
                       get_all: Optional[bool] = False) -> Union[str, Iterable[str]]:
        r"""Get data by globbing ID from the basename of files.

        Args:
            id (str):
                The ID of the desired data.
            globber (str, Optional):
                Regex pattern to glob ID from the loaded files. If `None`, the stored attribute
                :attribute:`_id_globber` will be used.
            get_all (bool, Optional):
                If ``True``, get all the data with the same IDs if muiltiple instances were
                identified by the same ID. Default to ``False``.

        Return:
            torch.Tensor or list

        """
        if globber is None:
            globber = self._id_globber

        ids = self.get_unique_IDs(globber)
        if len(set(ids)) != len(ids) and not get_all:
            self._logger.warning("IDs are not unique using this globber: %s!"%globber)

        if ids.count(id) <= 1 or not get_all:
            return self.__getitem__(ids.index(id))
        else:
            self._logger.warning(f"Returning first that matches requested ID {id}. "
                                 f"{[self.get_data_source(i) for i in np.where(np.array(ids)==id)[0]]}")
            return [self.__getitem__(i) for i in np.where(np.array(ids)==id)[0]]

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
        import re

        if not globber is None:
            self._id_globber = globber
        filenames = [os.path.basename(self.get_data_source(i)) for i in range(self.__len__())]

        outlist = []
        for f in filenames:
            matchobj = re.search(self._id_globber, f)

            if not matchobj is None:
                outlist.append(f[matchobj.start():matchobj.end()])
        if len(set(outlist)) !=len(outlist):
            duplicate = pd.Series(data=outlist, index=outlist).duplicated()
            self._logger.warning(f"Some IDs are not unique: \n{duplicate.to_string()}")
        return outlist

    def get_size(self, i: int) -> Iterable[int]:
        r"""Get the size of the original image. Gives 3D size.

        Args:
            i (int): Index.

        Returns:
            Iterable[int]
        """
        i = i % len(self.metadata)
        return [int(self.metadata[i]['dim'][j + 1]) for j in range(3)]

    def get_spacing(self, i: int) -> Iterable[float]:
        r"""Get the spacing of the original image. Ignores load by slice and
        gives 3D spacing. Note that the output is rounded to 8-th decimal place

        Args:
            i (int): Index.

        Returns:
            Iterable[float]: Spacing in mm.
        """
        i = i % len(self.metadata)
        return [round(self.metadata[i]['pixdim'][j + 1], 8) for j in range(3)]

    def get_origin(self, i: int) -> Iterable[float]:
        r"""Get the origin of the image. Note that the output is rounded to the third decimal
        place

        Args:
            i (int): Index.

        Returns:
            Iterable[float]: Physical coordinates extracted from q-form matrix.


        """
        origin = [round(self.metadata[i][k], 3) for k in ['qoffset_x',
                                                          'qoffset_y',
                                                          'qoffset_z']]
        return origin

    def get_direction(self, i: int) -> Iterable[float]:
        r"""Get the orientation of the image. Note that the output is rounded to the third
        decimal place.

        Args:
            i (int): Index.

        Returns:
            Iterable[float]: Affine direction defined by quartern vector.

        See Also:
            http://learningnotes.fromosia.com/index.php/2017/03/10/image-orientation-vtk-itk/

        """
        direction = [round(self.metadata[i][k], 3) for k in ['quatern_b',
                                                             'quatern_c',
                                                             'quatern_d']]


        return direction

    def get_properties(self, i: int) -> dict:
        r"""Get the properties of the target data inlucing spacing, orientation, origin, dimension...etc
        
        Args:
            i (int): Index.

        Return:
            dict: Contains keys {'size', 'spacing', 'origin', 'direction'}.
        """
        i = i % len(self.metadata)

        size = self.get_size(i)
        spacing = self.get_spacing(i)
        origin = self.get_origin(i)
        direction = self.get_direction(i)
        return {'size': size,
                'spacing': spacing,
                'origin': origin,
                'direction': direction}

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> torch.Tensor:
        for r in range(len(self)):
            yield self.data[r]

    def __getitem__(self, item) -> torch.Tensor:
        r"""Override to generate tensor.

        Args:
            item (Any):
                ID to get the item desired.

        Returns:
            ``torch.Tensor``
        """
        out = self.data[item][tio.DATA]
        return out

    def __str__(self):
        s = "==========================================================================================\n" \
            "Datatype: %s \n" \
            "Root Path: %s \n" \
            "Number of loaded images: %i\n" \
            "Image Details:\n" \
            "--------------\n"%(__class__.__name__, self.rootdir, self.length)
        # "File Paths\tSize\t\tSpacing\t\tOrigin\n"
        # printable = {'File Name': []}
        if self.metadata_table is None:
            self.update_metadata_table()
        s += self.metadata_table.to_string()
        return s

    def update_metadata_table(self) -> pd.DataFrame:
        r"""
        Populate self.metadata_table
        """
        from pandas import DataFrame as df
        printable = {'ID': [], 'File Name': [], 'Size': [], 'Spacing': [], 'Origin': [], 'Orientation': []}
        for i in range(len(self.data_source_path)):
            id_mo = re.search(self._id_globber, os.path.basename(self.data_source_path[i]))
            id_mo = 'None' if id_mo is None else id_mo.group()
            printable['ID'].append(id_mo)
            printable['File Name'].append(os.path.basename(self.data_source_path[i]))

            # TODO: temp fix
            printable['Size'].append(self.metadata[i]['dim'][1:4])
            printable['Spacing'].append([round(self.metadata[i]['pixdim'][j], 2) for j in range(1, 4)])
            printable['Origin'].append([round(self.metadata[i][k], 3) for k in ['qoffset_x',
                                                                                'qoffset_y',
                                                                                'qoffset_z']])
            printable['Orientation'].append(self.metadata[i]['orientation'])
        data = df.from_dict(data=printable)
        data = data.set_index('ID')
        self.metadata_table = data
        return data

    def write_all(self,
                  tensor_data: torch.Tensor,
                  outputdirectory: str,
                  prefix: Optional[str] = '') -> None:
        r"""Write data array to the output directory according to the image
        properties of the loaded images.

        Args:
            tensor_data (:obj:`torch.tensor`):
                Data arrays to save, has to be arranged identically as the attribute self.data
                of the object. Expect dimension :math:`(B × C × D × W × H)`
            outputdirectory (str):
                Folder to output nii files.
            prefix (str):
                Prefix to add before saved files. Default to ''.
        """
        for i in range(len(self)):
            source_file = self.data_source_path[i]
            self.write_uid(tensor_data[i].squeeze().numpy(), i, outputdirectory, prefix)

    def write_uid(self,
                  tensor_data: torch.Tensor,
                  unique_id: str,
                  outputdirectory: str,
                  prefix: Optional[str] = '') -> None:
        r"""Write data with reference to the source image with specified unique_id.

        Args:
            tensor_data (torch.Tensor):
                Data array to save, should have dimension :math:`(D × W × H)`
            unique_id (Any):
                If str, source image with same unique ID is loaded. If int, source image load at
                the same index in `data_source_path` is loaded.
            outputdirectory (str):
                Folder to output the nii files
            prefix (str, Optional):
                Prefix to add before the saved files. Default to ''.
            suffix:
                Prefix to add after the saved files. Default to ''.
        """

        # Load source image
        if isinstance(unique_id, str):
            index = self.get_unique_IDs().index(str)
        elif isinstance(unique_id, int):
            index = unique_id
        else:
            raise TypeError(f"Incorrect unique id specified, expect [int or str] got {unique_id}")

        src_path = self.data_source_path[index]
        src_im = sitk.ReadImage(src_path)
        out_im = sitk.GetImageFromArray(tensor_data.squeeze().numpy())

        # Check if size equal
        assert src_im.GetSize() == out_im.GetSize(), f"Source image and target image has different sizes: " \
                                                     f"\tsource: {src_im.GetSize()}\ttarget: {out_im.GetSize()}"

        out_im.CopyInformation(src_im)
        out_name = outputdirectory +'/' + prefix + os.path.basename(self.data_source_path[index])
        self._logger.info(f"Writing {out_name}")
        sitk.WriteImage(out_im, out_name)

    def get_unique_values(self) -> Any:
        r"""Get the tensor of all unique values in basedata. Only for integer tensors.
        """
        assert self[0].is_floating_point() == False, \
            "This function is for integer tensors. Current datatype is: %s"%(self[0].dtype)
        vals = unique(cat([unique(d) for d in self]))
        return vals

    def get_unique_values_n_counts(self) -> dict:
        """Get a dictionary of unique values as key and its counts as value.

        Returns:
            dict: Key value pairs of unique values and their counts.
        """
        from torch.utils.data import DataLoader
        assert self[0].is_floating_point() == False, \
            "This function is for integer tensors. Current datatype is: %s"%(self[0].dtype)

        out_dict = {}

        # torchio reuqires some tricks to keep memory efficiencies.
        subjects = [tio.Subject(im=d) for d in self.data]
        subjects = tio.SubjectsDataset(subjects)
        subjects_loader = DataLoader(subjects, batch_size=1, num_workers=12)

        # Use a dataloader to do the trick
        for d in tqdm(subjects_loader, desc="get_unique_values_n_counts"):
            val, counts = unique(d['im'][tio.DATA], return_counts=True)
            for v, c, in zip(val, counts):
                if v.item() not in out_dict:
                    out_dict[v.item()] = c.item()
                else:
                    out_dict[v.item()] += c.item()
            del d
        return out_dict

