from torch import from_numpy, cat, stack, unique
from torch.nn.functional import pad
from .PMIDataBase import PMIDataBase
from tqdm import *
import tqdm.auto as auto
import fnmatch, re
import os
import numpy as np
import SimpleITK as sitk


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
            Decide image directories globbing method, whether to look into subdirectories or not. \n
            Possible values:
                * `normal` - typical loading behavior, reading all nii/nii.gz files in the directory.
                * `recursive` - search all subdirectories excluding softlinks, use with causion.
                * `explicit` - specifying directories of the files to load.
            Default is `normal`.
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

        Loading images:

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


        Geting image from object:

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
    """
    def __init__(self, rootdir, readmode='normal', filtermode=None, loadBySlices=-1, verbose=False, dtype=float,
                 debugmode=False, **kwargs):
        super(ImageDataSet, self).__init__(verbose=verbose)
        assert os.path.isdir(rootdir), "Cannot access directory: {}".format(rootdir)
        assert loadBySlices <= 2, "This class only handle 3D data!"

        self.rootdir            = rootdir
        self.data_source_path   = []
        self.data               = []
        self.metadata           = []
        self.length             = 0
        self.verbose            = verbose
        self.dtype              = dtype
        self._raw_length        = 0 # length of raw input (i.e. num of nii.gz files loaded)
        self._filterargs        = kwargs
        self._filtermode        = filtermode
        self._readmode          = readmode
        self._id_globber        = kwargs.get('idGlobber', "(^[a-zA-Z0-9]+)")
        self._debug             = debugmode
        self._byslices          = loadBySlices

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
            assert all([ k in self._filterargs for k in ['idlist', 'regex']]), 'No filter arguments.'


    def _parse_root_dir(self):
        r"""
        Main parsing function.
        """

        self._logger.info("Parsing root path: " + self.rootdir)

        #===================================
        # Read all nii.gz files exist first.
        #-----------------------------------
        removed_fnames = []
        if self._readmode is 'normal':
            file_dirs = os.listdir(self.rootdir)
            file_dirs = fnmatch.filter(file_dirs, "*.nii.gz")
            file_dirs = [os.path.join(self.rootdir, f) for f in file_dirs]
        elif self._readmode is 'explicit':
            file_dirs = [fs.rstrip() for fs in open(self.rootdir, 'r').readlines()]
            for fs in file_dirs:
                if not os.path.isfile(fs):
                    file_dirs.remove(fs)
                    removed_fnames.append(fs)
        elif self._readmode is 'recursive':
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
        for i, f in enumerate(auto.tqdm(file_dirs, disable=not self.verbose, desc="Load Images")) \
                if not self._debug else enumerate(auto.tqdm(file_dirs[:10],
                                                       disable=not self.verbose,
                                                            desc="Load Images")):
            if self.verbose:
                self._logger.info("Reading from "+f)
            im = sitk.ReadImage(f)
            self.data_source_path.append(f)
            imarr = sitk.GetArrayFromImage(im).astype(self.dtype)
            self.data.append(from_numpy(imarr))
            self._itemindexes.append(self.data[i].size()[0])
            metadata = {}
            for key in im.GetMetaDataKeys():
                try:
                    if key.split('['):
                        key_type = key.split('[')[0]

                    try:
                        t = NIFTI_DICT[key_type]
                    except:
                        continue
                    metadata[key] = t(im.GetMetaData(key))
                except:
                    metadata[key] = im.GetMetaData(key)
            self.metadata.append(metadata)
            self._raw_length += 1
        self.length = len(self.data_source_path)
        self._logger.info("Finished loading. Loaded {} files.".format(self.length))

        #=====================================
        # Option to load 3D images as 2D slice
        #-------------------------------------
        if self._byslices >= 0:
            try:
                self._logger.info("Load by slice...")
                self._itemindexes = np.cumsum(self._itemindexes)
                self.length = np.sum([m.size()[self._byslices] for m in self.data])
                # check if all sizes are the same
                allsizes = [tuple(np.array(m.size())[np.arange(m.dim()) != self._byslices]) for m in self.data]
                uniquesizes = list(set(allsizes))
                if not len(uniquesizes) == 1:
                    self._logger.debug("Detected slice sizes: {}".format(uniquesizes))
                    self._logger.warning("There are more than one slice size, attempting to pad/crop")
                    majority_size = uniquesizes[np.argmax([allsizes.count(tup) for tup in uniquesizes])]
                    self._logger.info("Found majority size: {}".format(majority_size))

                    # Get all index of image that is not of majority size
                    target = [ss != majority_size for ss in allsizes]
                    target = [i for i, x in enumerate(target) if x]
                    self._logger.info("Targets that are not of majority size: {}".format(target))

                    self._crop_data(target, majority_size, allsizes)

                self.data = cat(self.data, dim=self._byslices).transpose(0, self._byslices).unsqueeze(1)
                self._logger.info("Finished load by slice.")
            except IndexError:
                self._logger.warning("Wrong Index is used in load by slices option!")
                self._logger.warning("Retreating...")
                self.length = len(self.data_source_path)
        else:
            try:
                self.data = stack(self.data, dim=0).unsqueeze(1)
                self._logger.debug(f"self.data.shape:{self.data.shape}")
            except:
                self._logger.warning("Cannot stack data due to non-uniform shapes.")
                self._logger.debug("Shapes are: \n%s"%'\n'.join([str(d.shape) for d in self.data]))
                self._logger.warning("Some function might be impaired. Trying to unify size!")

                allsizes = [tuple(m.shape[1:]) for m in self.data]
                uniquesizes = list(set(allsizes))
                self._logger.info("Dectected image sizes: {}".format(uniquesizes))

                # Make 3D data into images with the same slice but not forcing them to have same number of slices
                if not len(uniquesizes) == 1:
                    majority_size = uniquesizes[np.argmax([allsizes.count(tup) for tup in uniquesizes])]
                    self._logger.info("Found majority size: {}".format(majority_size))
                    target = [ss != majority_size for ss in allsizes]
                    target = [i for i, x in enumerate(target) if x]
                    self._logger.debug("Reisize needed for: {}".format([self.get_unique_IDs()[t]
                                                                       for t in target]))

                    self._crop_data(target, majority_size, allsizes)

    def _crop_data(self, target_images, target_size, allsizes):
        r"""Crop the images into one with equal X-Y dimension. Used in `parse_root_dir`."""
        for t in target_images:
            self._logger.info("Trying to pad/crop {}".format(t))
            target_dat = self.data[t]
            target_size = np.array(target_size)
            tmp_im_shape = np.array(target_dat.shape)
            pad_size_left = (target_size - tmp_im_shape[-2:]) // 2
            pad_size_right = target_size - pad_size_left - tmp_im_shape[-2:]

            # Check if majority size is greater than original size.
            Pad = target_size[-2:] < np.array(target_size)

            self._logger.debug("Current size: {}".format(tmp_im_shape))
            self._logger.debug("Target size: {}".format(target_size))
            self._logger.debug("pad_size_left/right: {},{}".format(pad_size_left,
                                                                   pad_size_right))

            # Check if majority size is greater than original size.
            need_pad = target_size[-2:] > tmp_im_shape[-2:]

            # Crop or pad image to standard size
            for dim, (ps_l, ps_r, p) in enumerate(zip(pad_size_left, pad_size_right, need_pad)):
                t_dim = dim  # we are only doing H W padding, not Z
                if not p:
                    self._logger.debug(f"(ps_l={ps_l}, ps_r={ps_r}, t_dim={t_dim}, target_size={target_size}"
                                       f"(target_dat.shape={target_dat.shape}")
                    target_dat = target_dat.narrow(int(t_dim), int(abs(ps_l)), int(target_size[dim]))
                else:
                    pa = [0] * target_dat.ndim * 2
                    pa[t_dim * 2] = abs(int(ps_l))
                    pa[t_dim * 2 + 1] = abs(int(ps_r))
                    self._logger.debug("Padding: {}".format(pa))
                    target_dat = pad(target_dat, pa[-4:], mode='constant', value=0)

            self._logger.debug("Resized {} from {} to {}".format(
                self.get_unique_IDs()[t],
                allsizes[t],
                list(target_dat.shape)
            ))

            # Make sure channel dimension is there
            while target_dat.dim() < 4:
                target_dat = target_dat.unsqueeze(0)
            self.data[t] = target_dat

    def _filter_filelist(self, file_dirs, filtered_away, removed_fnames):
        r"""Filter the `file_dirs` using the specified attributions. Used in `parse_root_dir`."""
        # Filter by filelist
        if self._filtermode == 'idlist' or self._filtermode == 'both':
            self._logger.info("Globbing ID with globber: " + self._id_globber + " ...")
            file_basenames = [os.path.basename(f) for f in file_dirs]
            file_ids = [re.search(self._id_globber, f) for f in file_basenames]
            file_ids = [str(mo.group()) if not mo is None else mo for mo in file_ids]

            if isinstance(self._filterargs['idlist'], str):
                self._idlist = [r.strip() for r in open(self._filterargs['idlist'], 'r').readlines()]
            elif self._filterargs['idlist'] is None:
                self._logger.warning('Idlist input is None!')
                pass
            else:
                self._idlist = self._filterargs['idlist']

            self._logger.debug(f'{self._idlist}')
            tmp_file_dirs = np.array(file_dirs)
            keep = [id in self._idlist for id in file_ids]  # error near this could be because nothing is grabed

            file_dirs = tmp_file_dirs[keep].tolist()
            filtered_away.extend(tmp_file_dirs[np.invert(keep)])

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
            elif self._filterargs['regex'][0] == '(':
                try:
                    keep = np.invert([re.match(self._filterargs['regex'], f) is None for f in file_basenames])
                except Exception as e:
                    import sys, traceback as tr
                    cl, exc, tb = sys.exc_info()
                    self._logger.error("Error encountered when performing regex filtering.")
                    self._logger.error(tr.extract_tb(tb))
                    self._logger.error([re.match(self._filterargs['regex'], f) is None for f in file_basenames])

                try:
                    filtered_away.extend(np.array(file_dirs)[np.invert(keep)].tolist())
                except IndexError:
                    self._logger.exception("Error when trying to filter by regex.")
                    self._logger.debug(f"keep: {keep}")
                    self._logger.debug(f"file_dirs: {file_dirs}")
                except:
                    self._logger.exception("Unknown error when trying to filter by regex.")
                file_dirs = np.array(file_dirs)[keep].tolist()
            else:  # else use wild card
                file_dirs = fnmatch.filter(file_dirs, "*" + self._filterargs['regex'] + "*")
        if len(removed_fnames) > 0:
            removed_fnames.sort()
            for fs in removed_fnames:
                self._logger.warning("Cannot find " + fs + " in " + self.rootdir)
        file_dirs.sort()
        return file_dirs

    def get_raw_data_shape(self):
        r"""Get shape of all files as a list (ignore load by slice option)."""
        return [self.get_size(i) for i in range(len(self.metadata))]

    def check_shape_identical(self, target_imset):
        r"""Check if file shape is identical to another ImageDataSet."""
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

    def size(self, i=None):
        r"""Required by pytorch."""
        if i is None:
            try:
                return self.data.shape
            except:
                return self.length
        else:
            return self.length

    def type(self):
        r"""Return datatype of the elements."""
        return self.data[0].type()

    def as_type(self, t):
        r"""Cast all elements to specified type."""
        try:
            self.data = self.data.type(t)
            self.dtype = t
        except Exception as e:
            self._logger.error("Error encounted during type cast.")
            self._logger.log_traceback()

    def get_data_source(self, i):
        r"""Get directory of the source of the i-th element."""
        if self._byslices >=0:
            try:
                return self.data_source_path[int(np.argmax(self._itemindexes > i)) - 1]
            except IndexError:
                # self._logger.warning("Require index {} but source path len is {}.".format(
                #     int(np.argmax(self._itemindexes > i)) - 1,
                #     len(self.data_source_path)
                # ))
                # self._logger.warning("Returning modulated result.")
                return self.data_source_path[(int(np.argmax(self._itemindexes > i)) - 1) % len(self.data_source_path)]
        else:
            return self.data_source_path[i]

    def get_internal_index(self, i):
        r"""If load by slice, get the image index instead of the stacked slice index."""
        if self._byslices >= 0:
            return int(np.argmax(self._itemindexes > i)) - 1
        else:
            return i

    def get_internal_slice_index(self, i):
        r"""If load by slice, get the slice number of the i-th 2D element in
        its original image."""
        if self._byslices >= 0:
            return i - self._itemindexes[int(np.argmax(self._itemindexes > i)) - 1]
        else:
            return i

    def get_data_by_ID(self, id, globber=None, get_all=False):
        r"""Get data by globbing ID from the basename of files."""
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

    def get_unique_IDs(self, globber=None):
        r"""Get all IDs globbed by the specified globber. If its None,
        default globber used. If its not None, the class globber will be
        updated to the specified one.
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
        return outlist

    def get_size(self, id: int):
        r"""Get the size of the original image. Gives 3D size. If load by slices, it will look for the internal
        index before returning the 3D size.
        """
        if self._byslices >= 0:
            return [int(self.metadata[self.get_internal_index(id)]['dim[%d]' % (i + 1)]) for i in range(3)]
        else:
            id = id % len(self.metadata)
            return [int(self.metadata[id]['dim[%d]'%(i+1)]) for i in range(3)]

    def get_spacing(self, id):
        r"""Get the spacing of the original image. Ignores load by slice and
        gives 3D spacing."""
        id = id % len(self.metadata)
        if self._byslices >= 0:
            return [round(self.metadata[self.get_internal_index(id)]['pixdim[%d]' % (i + 1)], 5) for i in range(3)]
        else:
            return [round(self.metadata[id]['pixdim[%d]'%(i+1)], 5) for i in range(3)]

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self._byslices >= 0:
            out_dim = 3
        else:
            out_dim = 4

        out = self.data[item]
        while out.dim() < out_dim:
            out = out.unsqueeze(0)
        return out

    def __str__(self):
        from pandas import DataFrame as df
        s = "==========================================================================================\n" \
            "Datatype: %s \n" \
            "Root Path: %s \n" \
            "Number of loaded images: %i\n" \
            "Load By Slice: %i \n" \
            "Image Details:\n" \
            "--------------\n"%(__class__.__name__, self.rootdir, self.length, self._byslices)
        # "File Paths\tSize\t\tSpacing\t\tOrigin\n"
        # printable = {'File Name': []}
        printable = {'ID': [], 'File Name': [], 'Size': [], 'Spacing': [], 'Origin': []}
        for i in range(len(self.data_source_path)):
            id_mo = re.search(self._id_globber, os.path.basename(self.data_source_path[i]))
            id_mo = 'None' if id_mo is None else id_mo.group()
            printable['ID'].append(id_mo)
            printable['File Name'].append(os.path.basename(self.data_source_path[i]))
            # for keys in self.metadata[i]:
            #     if not printable.has_key(keys):
            #         printable[keys] = []
            #
            #     printable[keys].append(self.metadata[i][keys])
            printable['Size'].append([self.metadata[i]['dim[1]'],
                                      self.metadata[i]['dim[2]'],
                                      self.metadata[i]['dim[3]']])
            printable['Spacing'].append([round(self.metadata[i]['pixdim[1]'], 2),
                                         round(self.metadata[i]['pixdim[2]'], 2),
                                         round(self.metadata[i]['pixdim[3]'], 2)])
            printable['Origin'].append([round(self.metadata[i]['qoffset_x'], 2),
                                        round(self.metadata[i]['qoffset_y'], 2),
                                        round(self.metadata[i]['qoffset_z'], 2)])
        data = df(data=printable)
        data = data.set_index('ID')
        s += data.to_string()
        return s

    def Write(self, tensor_data, outputdirectory, prefix=''):
        r"""Write data array to the output directory accordining to the image
        properties of the loaded images.

        Args:
            tensor_data (:obj:`torch.tensor`):
                Data arrays to save, has to be arranged identically as the attribute self.data
                of the object.
            outputdirectory (str):
                Folder to output nii files.
            prefix (str):
                Prefix to add before saved files. Default to ''.
        """
        if self._byslices > -1:
            assert self._itemindexes[-1] == tensor_data.size()[0], \
                "Dimension mismatch! (%s vs %s)"%(self._itemindexes[-1], tensor_data.size()[0])
            td=tensor_data.numpy()
            for i in range(len(self.data_source_path)):
                self._logger.info("Writing for {} with source image: {}".format(self.get_unique_IDs()[self._itemindexes[i]],
                                                                                self.data_source_path[i]))
                start=self._itemindexes[i]
                end=self._itemindexes[i+1]
                # image=sitk.GetImageFromArray(td[start:end])
                templateim = sitk.ReadImage(self.data_source_path[i])

                # check if it matches the original image size
                tmp_im = td[start:end]
                if not np.roll(tmp_im.shape, -1).tolist() == list(templateim.GetSize()):
                    self._logger.info("Recovering size for image with ID: {}.".format(
                        self.get_unique_IDs()[self._itemindexes[i]]))

                    target_dat = templateim
                    target_size = np.array(target_dat.GetSize())[:2]
                    tmp_im_shape = np.array(tmp_im.shape)
                    pad_size_left = (target_size - tmp_im_shape[-2:]) // 2
                    pad_size_right = target_size - pad_size_left - tmp_im_shape[-2:]

                    self._logger.debug("Current size: {}".format(tmp_im.shape))
                    self._logger.debug("Target size: {}".format(target_size))

                    # Check if majority size is greater than original size.
                    need_pad = target_size[-2:] > tmp_im_shape[-2:]

                    tmp_im = from_numpy(tmp_im)
                    # Crop or pad image to standard size
                    for dim, (ps_l, ps_r, p) in enumerate(zip(pad_size_left, pad_size_right, need_pad)):
                        t_dim = dim + 1 if dim >= self._byslices else dim # since by slice eats a dimension
                        if not p:
                            tmp_im = tmp_im.narrow(int(t_dim), int(ps_l), int(tmp_im_shape[dim]))
                        else:
                            pa = [0] * tmp_im.ndim * 2
                            pa[t_dim * 2] = abs(int(ps_l))
                            pa[t_dim * 2 + 1] = abs(int(ps_r))
                            self._logger.debug("Padding: {}".format(pa))
                            # pa = [abs(int(ps_l)) if x // 2 == t_dim else 0 for x in range(6)]
                            # if len(pa) == 4: (left, right, top, bot)
                            # if len(pa) == 6: (left, right, top, bot, front, back)
                            tmp_im = pad(tmp_im, pa[-4:], mode='constant', value=0)
                    tmp_im = tmp_im.numpy()

                    self._logger.info("Resized to shape: {}".format(tmp_im.shape))


                image = sitk.GetImageFromArray(tmp_im)
                image.CopyInformation(templateim)
                # image=self.WrapImageWithMetaData(td[start:end], self.metadata[i])
                sitk.WriteImage(image, outputdirectory +'/' + prefix + os.path.basename(self.data_source_path[i]))
                del tmp_im

        else:
            assert len(self) == len(tensor_data), "Length mismatch! %i vs %i"%(len(self), len(tensor_data))

            for i in range(len(self)):
                source_file = self.data_source_path[i]
                templateim = sitk.ReadImage(source_file)
                image = sitk.GetImageFromArray(tensor_data[i].squeeze().numpy())
                image.CopyInformation(templateim)
                sitk.WriteImage(image, outputdirectory +'/' + prefix + os.path.basename(self.data_source_path[i]))

    def get_unique_values(self):
        r"""Get the tensor of all unique values in basedata. Only for integer tensors
        """
        assert self.data[0].is_floating_point() == False, \
            "This function is for integer tensors. Current datatype is: %s"%(self.data[0].dtype)
        vals = unique(cat([unique(d) for d in self.data]))
        return vals

    def get_unique_values_n_counts(self):
        """Get a dictionary of unique values as key and its counts as value.
        """
        assert self.data[0].is_floating_point() == False, \
            "This function is for integer tensors. Current datatype is: %s"%(self.data[0].dtype)
        out_dict = {}
        for val, counts in [unique(d, return_counts=True) for d in self.data]:
            for v, c in zip(val, counts):
                if v.item() not in out_dict:
                    out_dict[v.item()] = c.item()
                else:
                    out_dict[v.item()] += c.item()
        return out_dict

