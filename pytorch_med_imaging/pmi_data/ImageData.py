import ast
from typing import Any, Iterable, Optional, Union, Tuple

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
from pathlib import Path

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
    r"""ImageDataSet class that reads and load nifty in a specified directory.

    This class loads .nii.gz files from a specified directory and assign ID to each of the loaded files
    based on a regex pattern. This class also loads header information, such as origin, direction of the
    image files, which can be used to write the image with the same header information but different
    data.

    Iterating this class returns either a torchio.ScalarImage or torchio.LabelImage image depending on
    the `dtype` argument.

    .. hint::
        The ID globber is expected to glob unique IDs from the input file names. If it's not unique
        and `raise_id_duplicate` is `False`, the first of the duplicat is kept.

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
            id_globber. See formats in notes.
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

            >>> from pytorch_med_imaging.pmi_data import ImageDataSet
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


        **Geting torch tensor image from object:**

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

        **Getting torchio image object**

        >>> imsset.data[id]

    .. hint::
        Use ``instance[item]`` to get the data as ``torch.Tensor``.

    .. note::
        The format of argument ``idlist`` should be one of
        * ``str``: if it is a string, it should either be a directory to a txt/ini file or a plaintext list of comma
        separated values, which also needed to be wrapped with square brackets. ``ast.litera_eval()`` will be used to
        convert the string into a list.
        * ``list``: if it is a list, the list will be copied directly as the targeted ids
        * ``tuple``: same as `list` inputs.
    """
    def __init__(self, rootdir, readmode='normal', filtermode=None, verbose=False, dtype=float,
                 debugmode=False, raise_id_duplication=False, **kwargs):
        super(ImageDataSet, self).__init__(verbose=verbose)
        self.rootdir: Path      = Path(rootdir)
        self.metadata           = []
        self.metadata_table     = None
        self.verbose            = verbose
        self._dtype             = dtype # This is the desired type
        self._data              = pd.Series(name="Data")
        self._raw_length        = 0 # length of raw input (i.e. num of nii.gz files loaded)
        self._filterargs        = kwargs
        self._filtermode        = filtermode
        self._readmode          = readmode
        self._id_globber        = kwargs.get('id_globber', "(^[a-zA-Z0-9]+)")
        self._debug             = debugmode
        self._raise_id_dup      = raise_id_duplication

        assert self.rootdir.is_dir(), f"Specified rootdir {rootdir} doesn't exist."
        self._error_check()
        self._parse_root_dir()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __iter__(self) -> Iterable[tio.Image]:
        for r in range(len(self)):
            yield self.data.iloc[r]

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

    def iter_torch_tensor(self) -> Iterable[torch.Tensor]:
        for r in range(len(self)):
            if isinstance(item, slice):
                out = torch.cat([d[tio.DATA] for d in out.values])
            elif isinstance(out, (tio.Image, tio.ScalarImage, tio.LabelMap)):
                out = out[tio.DATA]
            yield out

    @property
    def length(self):
        return len(self)

    @property
    def data_source_path(self):
        return self.metadata['Path']


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

        if not isinstance(self._id_globber, str):
            raise ArgumentError("id_globber must be specified and must be string.")

    def _parse_root_dir(self):
        r"""
        Main parsing function.
        """
        self._logger.info("Parsing root path: " + str(self.rootdir))

        #===================================
        # Read all nii.gz files exist first.
        #-----------------------------------
        if self._readmode == 'normal':
            file_dirs = self.rootdir.glob("*nii.*")
        elif self._readmode == 'explicit' and self.rootdir.suffix == '.txt':
            file_dirs = [fs.rstrip() for fs in open(self.rootdir, 'r').readlines()]
            for fs in file_dirs:
                if not os.path.isfile(fs):
                    file_dirs.remove(fs)
        elif self._readmode == 'recursive':
            file_dirs = self.rootdir.rglob("*nii.*")
        else:
            raise AttributeError("file_dirs is not assigned!")
        file_dirs = list(file_dirs)

        if len(file_dirs) == 0:
            self._logger.error("No target files found in {}.".format(self.rootdir))
            raise ArithmeticError("No target files found in {}.".format(self.rootdir))
        else:
            # establish the uid mapping
            file_map = []
            for f in file_dirs:
                mo = re.search(self._id_globber, f.name)
                if not mo is None:
                    file_map.append(pd.Series({'UID': mo.group(),
                                     'Path': f}))
                else:
                    self._logger.warning(f"File does not have an ID: {f.name}")
            data_source_path = pd.concat(file_map, axis=1).T
            data_source_path.set_index("UID", inplace=True, drop=True)

        # Check if there's any duplicated index
        if data_source_path.index.duplicated().any():
            duplicated = data_source_path[data_source_path.index.duplicated(keep='first')]
            self._logger.warning(f"Duplicated entries found in data_source_path! Dropping: \n{duplicated}")
            if self._raise_id_dup:
                raise KeyError("ID globber does not lead to unique IDs. Clean the source directory!")
            else:
                data_source_path = data_source_path[~data_source_path.index.duplicated(keep='first')]

        #==========================
        # Apply filter if specified
        #--------------------------
        data_source_path = self._filter_filelist(data_source_path) # Note this might update self._data
        self._logger.info("Found %s nii.gz files..."%len(data_source_path))
        self._logger.info("Start Loading")

        #=============
        # Reading data
        #-------------
        for k, f in tqdm(data_source_path.iterrows(), disable=not self.verbose, desc="Load Images"):
            f = f[0]
            if self._debug and i >= 10:
                break

            if self.verbose:
                self._logger.info(f"Reading from {str(f)}")

            if not os.path.isfile(f):
                self._logger.warning("Cannot find file!")
                self._logger.debug(f"{os.listdir(f.parent)}")

            # if dtype is uint, treat as label
            if np.issubdtype(self._dtype, np.unsignedinteger):
                im = tio.LabelMap(f)
            else:
                im = tio.ScalarImage(f, check_nans=True)
            self._data[k] = [] # Pandas will try to cast the dtype, this prevents it
            self._data[k] = im

            # read metadata
            nib_im = nib.load(str(f))
            im_header = nib_im.header
            im_header_dict = {key: im_header.structarr[key].tolist() for key in im_header.structarr.dtype.names}
            im_header_dict['orientation'] = im.orientation
            im_header_dict['UID'] = k
            self.metadata.append(pd.Series(im_header_dict))
        self.metadata = pd.concat(self.metadata, axis=1).T
        self.metadata.set_index("UID", inplace=True, drop=True)
        self.metadata = self.metadata.join(data_source_path)

        self._logger.info("Finished loading. Loaded {} files.".format(self.length))
        self._logger.debug(f"IDs of loaded images: {','.join(self.get_unique_IDs())}")

    def _filter_filelist(self, file_map: pd.DataFrame):
        r"""Filter the `file_dirs` using the specified attributions. Used in `parse_root_dir`."""
        # Filter by filelist
        #-------------------
        target_idlist = self._filterargs.get('idlist', None) or "" # required id list from arguments
        if (self._filtermode == 'idlist' or self._filtermode == 'both') and \
                target_idlist not in ("", None) and self._id_globber is not None:
            self._logger.info("Globbing ID with globber: " + str(self._id_globber) + " ...")
            file_ids = file_map.index

            if isinstance(target_idlist, str) and not target_idlist == "":
                target_idlist = target_idlist.strip('[]')
                if target_idlist.find(',') >= 0 and target_idlist is not None:
                    self._logger.info(f"Detect input as a list string, splitting at the commas.")
                    self._idlist = target_idlist.split(',')
                elif target_idlist.endswith(('.txt', '.ini')):
                    # If its a file directory
                    self._logger.info(f"Reading idlist from: {target_idlist}")
                    self._idlist = [r.strip() for r in open(target_idlist, 'r').readlines()]
            elif isinstance(target_idlist, (list, tuple)):
                # If its a list of IDs
                self._logger.info(f"Input idlist is already a list, directly using this list: "
                                  f"{target_idlist}")
                self._idlist = target_idlist
            elif target_idlist in (None, ""):
                # If None specified, glob ids from filenames instead
                self._logger.warning('Idlist input is None!')
                self._idlist = file_ids
            else:
                raise TypeError(f"ID list is not correclty spefified. Expect str, list or None, got "
                                f"{target_idlist} instead")

            # warn about discrepancy
            missing_ids = set(self._idlist) - set(file_ids)
            self._logger.debug(f'Target IDs: {self._idlist}')
            self._logger.debug(f'All globbed IDs: {file_ids}')
            if len(missing_ids):
                self._logger.warning(f"Missing ID(s): {set(self._idlist) - set(file_ids)}")

            # Finally, set the filemap
            file_map = file_map.loc[self._idlist]

            # Check if there are still things in the list
            if len(file_map) == 0:
                self._logger.warning("Nothing lefted in the file list after id-filtering! "
                                     "That can't be right, terminating.")
                raise FileNotFoundError("ID globber setting exclude all files.")

        # Fitlter by regex
        # --------------
        if self._filtermode == 'regex' or self._filtermode == 'both':
            self._logger.info("Filtering ID with filter: {}".format(self._filterargs['regex']))
            # use REGEX if find paranthesis
            if self._filterargs['regex'] is None:
                # do nothing if regex is Nonw
                self._logger.warning('Regex input is None, skipping')
                pass
            # if find *, treat it as wild card, if find .* treat it as regex
            elif self._filterargs['regex'].find('*') == -1 or self._filterargs['regex'].find('.*') > -1:
                try:
                    keep = pd.Series({k: re.match(self._filterargs['regex'], v.name) is None \
                                      for k, v in file_map['Path'].items()})
                    file_map = file_map.loc[keep[~keep.values].index]
                except Exception as e:
                    import sys, traceback as tr
                    self._logger.exception(e)
                    self._logger.debug(f"{file_map = }")
                    raise e
            else:  # else use wild card
                keep = fnmatch.filter(file_map.values, "*" + self._filterargs['regex'] + "*")
                keep = [f in keep for f in file_map.values]
                file_map = file_map.loc[keep]

            # Check if there are still things in the list
            if file_map.shape[0] == 0:
                self._logger.warning(
                    "Nothing lefted in the file list after regex-filtering! "
                    "That can't be right, terminating")
                raise FileNotFoundError("Regex setting exclude all files.")

        file_map.sort_index(inplace=True)
        return file_map

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

    def as_type(self, t) -> None:
        r"""Cast all elements to specified type."""
        raise DeprecationWarning("This is not functional anymore")

    def get_data_source(self, i) -> str:
        r"""Get directory of the source of the i-th element, sorted by filenames.

        Args:
            i (int): Index.

        Returns:
            str

        """
        if isinstance(i, int):
            return self.data_source_path.iloc[i]
        else:
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

        ids = self.get_unique_IDs()
        if len(set(ids)) != len(ids) and not get_all:
            self._logger.warning("IDs are not unique using this globber: %s!"%globber)

        if ids.count(id) <= 1 or not get_all:
            return self.__getitem__(ids.index(id))
        else:
            self._logger.warning(f"Returning first that matches requested ID {id}. "
                                 f"{[self.get_data_source(i) for i in np.where(np.array(ids)==id)[0]]}")
            return [self.__getitem__(i) for i in np.where(np.array(ids)==id)[0]]

    def get_data_as_tioimage(self,
                             item = Union[int, str],
                             get_all: Optional[bool] = False):
        if isinstance(item, str):
            # check if it's duplicated index
            if self._data.index.duplicated().loc[item]:
                if get_all:
                    self._logger.warning("Get_all option will be deprecated and all data must have unique ID. "
                                         "Multiple data instance should be created for items with same ID.",
                                         no_repeat=True)
                    return self._data.loc[item]
                else:
                    self._logger.warning(f"Returning first that matches requested ID {item}. ")
            else:
                return self._data.loc[item].values[0]
        else:
            return self._data.iloc[item].values[0]

    def get_data_as_torch_tensor(self, item):
        if isinstance(item, slice):
            raise KeyError("This function does not support slice input.")

        dat = self[item]
        return dat[tio.DATA]


    def get_size(self, i: int) -> Iterable[int]:
        r"""Get the size of the original image. Gives 3D size.

        Args:
            i (int): Index.

        Returns:
            Iterable[int]
        """
        i = i % len(self.metadata)
        return self.metadata.iloc[i]['dim'][1:4]

    def get_spacing(self, i: int) -> Iterable[float]:
        r"""Get the spacing of the original image. Ignores load by slice and
        gives 3D spacing. Note that the output is rounded to 8-th decimal place

        Args:
            i (int): Index.

        Returns:
            Iterable[float]: Spacing in mm.
        """
        i = i % len(self.metadata)
        return [round(self.metadata.iloc[i]['pixdim'][j + 1], 8) for j in range(3)]

    def get_origin(self, i: int) -> Iterable[float]:
        r"""Get the origin of the image. Note that the output is rounded to the third decimal
        place

        Args:
            i (int): Index.

        Returns:
            Iterable[float]: Physical coordinates extracted from q-form matrix.


        """
        origin = [round(self.metadata.iloc[i][k], 3) for k in ['qoffset_x','qoffset_y','qoffset_z']]
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
        direction = [round(self.metadata.iloc[i][k], 3) for k in ['quatern_b','quatern_c','quatern_d']]
        return direction

    def get_verbose_orientation(self, i: int) -> Tuple[str]:
        r"""Retrieve the detailed orientation of the image in verbose format (e.g., 'L', 'P', 'S').

        Args:
            i (int): Index of the image in the dataset.

        Returns:
            Tuple[str]: A tuple representing the orientation of the image.
        """
        orientation = self.metadata.iloc[i]['orientation']
        return orientation

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

    def get_raw_data_shape(self) -> list:
        r"""Get shape of all files as a list (ignore load by slice option).

        Returns:
            list[tuples]
        """
        return [self.get_size(i) for i in range(len(self.metadata))]

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

    def update_metadata_table(self) -> pd.DataFrame:
        r"""
        Populate self.metadata_table using :attr:`metadata` for a more readable view.
        """
        from pandas import DataFrame as df
        printable = pd.DataFrame(index=self.metadata.index)
        printable.join(self.metadata['Path'])
        printable['Spacing (mm)'] = self.metadata['pixdim'].apply(lambda x: [round(xx, 2) for xx in x[1:4]])
        printable['Origin'] = [[round(xx, 2) for xx in x] \
                               for _, x in self.metadata[['qoffset_x', 'qoffset_y', 'qoffset_z']].iterrows()]
        printable['Orientation'] = self.metadata.loc[printable.index]['orientation']
        self.metadata_table = printable
        return printable

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
                  prefix: Optional[str] = '', 
                  suffix: Optional[str] = '',
                  input_orientation: Optional[str] = None) -> None:
        r"""Write data with reference to the source image with specified unique_id. You may specify
        an orientation that represents the axis orientation of your input torch tensor.

        This method assumes all DICOM files are read via ITK and saved as Nifti, standardizing their
        orientation to LPS regardless of their original orientation. This typically works well, but
        issues may arise if DICOMs are loaded directly rather than Nifti files.

        .. note::
            The method expects tensors to maintain LPS orientation. If tensors are processed with external
            packages like `torchio.ToCanonical` or `nib` which may alter orientation, they should be
            adjusted back to LPS to ensure compatibility with this method. This function is designed to
            avoid axis swapping; it raises an error if the axis orientation does not match expected
            configurations. Orientation mismatches are handled simply by adjusting the sign, under the
            assumption that direction is the only discrepancy.


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

        Raise:
            NotImplementedError:
                Raised if the specified `input_orientation` require axis swapping when compared to the
                source image orientation (that is calculated with `GetDirection()` method).

        """

        # Load source image
        if isinstance(unique_id, str):
            index = self.get_unique_IDs().index(unique_id)
        elif isinstance(unique_id, int):
            index = unique_id
        else:
            raise TypeError(f"Incorrect unique id specified, expect [int or str] got {unique_id}")

        src_path = self.data_source_path[index]
        src_im = sitk.ReadImage(src_path)
        out_im = sitk.GetImageFromArray(tensor_data.squeeze().numpy())

        # Fix orientation
        if not input_orientation is None:
            if tuple(input_orientation) != self.metadata[index]['orientation']:
                # * Notes: See docstring notes for more details about this part
                src_ori = self.metadata[index]['orientation']
                inp_ori = tuple(input_orientation)
                self._logger.info(f"Detect orientation mismatch: src={src_ori} input={inp_ori}")
                self._logger.info(f"Reorientating {inp_ori} -> {src_ori}")
                # Make sure change does not involve axis swap
                allowed = ['LR', 'RL', 'AP', 'PA', 'SI', 'IS']
                for i_ori, s_ori in zip(inp_ori, src_ori):
                    if not i_ori + s_ori in allowed and i_ori != s_ori:
                        msg = (f"Correcting orientation {i_ori} to {s_ori} will involve axis swapping that is"
                               f" not supported.")
                        self._logger.warning(msg)
                        raise NotImplementedError(msg)
                # changing orientation now is just giving negative signs to the axis with mismatch direction
                self._logger.info(f"\tBefore: {out_im.GetDirection()}")
                out_im = sitk.DICOMOrient(out_im, ''.join(inp_ori))
                self._logger.info(f"\tAfter: {out_im.GetDirection()}")
                

        # Check if size equal
        if src_im.GetSize() != out_im.GetSize():
            msg = f"Source image and target image has different sizes: "\
                  f"\tsource: {src_im.GetSize()}\ttarget: {out_im.GetSize()}.\n" \
                  f"Make sure that patch sampling is not overlapped with transform `crop_or_pad` function."

        out_im.CopyInformation(src_im)
        if isinstance(outputdirectory, Path):
            outputdirectory = str(outputdirectory.absolute())

        base_name = re.sub('\.nii(\.gz)?}', '', os.path.basename(self.data_source_path[index]))
        out_name = outputdirectory +'/' + f"{prefix}{base_name}{suffix}.nii.gz"
        self._logger.info(f"Writing {out_name}")
        sitk.WriteImage(out_im, out_name)

    def sort_uid(self):
        r"""This is overriden because the data needs to maintain the same order as metadata."""
        super().sort_uid()
        self.metadata = self.metadata.loc[self._data.index]
        if self.metadata_table is not None:
            self.metadata_table = self.metadata_table.loc[self._data.index]
