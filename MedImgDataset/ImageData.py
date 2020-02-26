from torch.utils.data import Dataset
from torch import from_numpy, cat, tensor, stack, unique
from tqdm import *
import fnmatch, re
import os
import numpy as np
import SimpleITK as sitk

import logging


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

class ImageDataSet(Dataset):
    """ImageDataSet class that reads and load nifty in a specified directory

    Attributes
    ----------
    rootdir: str
        Path to the root directory for reading nifties
    readmode: str
        {'normal', 'recursive', 'explicit'}. Default is normal.
        Normal - typical loading behavior, reading all nii/nii.gz files in the directory.
        Recursive - search all subdirectories excluding softlinks, use with causion.
        Explicit - specifying directories of the files to load.
    filtermode: str
        {'idlist', 'regex', 'both', None}. Default is None.
        After grabbing file directories, they are filtered by either id or regex or both. Corresponding att needed.
    idlist: str or list
        If its str, it should be directory to a file containing IDs, one each line, otherwise, an explicit list.
        Need if filtermode is 'idlist'. Globber of id can be specified with attribute idGlobber.
    regex: str
        Regex that is used to match file directories. Un-matched ones are discarded. Effective when filtermode='idlist'
    idGlobber: str
        Regex string to search ID. Effective when filtermode='idlist', optional. If none specified the default
        globber is '(^[a-ZA-Z0-9]+), globbing the first one matches the regex in file basename. Must start with
        paranthesis.
    loadBySlices: int
        If its < 0, images are loaded as 3D volumes. If its >= 0, the slices along i-th dimension loaded.
    verbose: bool
        Whether to report loading progress.
    dtype: str or type
        Cast loaded data element to the specified type.
    debugmode:
        For debug only.
    recursiveSearch: bool
        Whether to load files recursively into subdirectories


    Examples
    --------
    Load all nii images in a folder:

    >>> from MedImgDataset import ImageDataSet
    >>> imgset = ImageDataSet('/some/dir/')

    Load all nii images, filtered by string 'T2W' in string:
    >>> imgset = ImageDataSet('/some/dir/', filtermode='regex', regex='(?=.*T2W.*)')

    Given a text file '/home/usr/loadlist.txt' with all image directories like this:
    /home/usr/img1.nii.gz
    /home/usr/img2.nii.gz
    /home/usr/temp/img3.nii.gz
    /home/usr/temp/img_not_wanted.nii.gz

    Load all nii images, filtered by file basename having numbers:
    >>> imgset = ImageDataSet('/home/usr/loadlist.txt', readmode='explicit')


    Get the first image:
    >>> im_1st = imgset[0]
    >>> im_2nd = imgset[1]
    >>> type(im_1st)
    torch.tensor


    """
    def __init__(self, rootdir, readmode='normal', filtermode=None, loadBySlices=-1, verbose=False, dtype=float,
                 debugmode=False, **kwargs):
        super(Dataset, self)
        assert os.path.isdir(rootdir), "Cannot access directory!"
        assert loadBySlices <= 2, "This class only handle 3D data!"
        self.rootdir = rootdir
        self.dataSourcePath = []
        self.data = []
        self.metadata = []
        self.length = 0
        self.verbose = verbose
        self.dtype = dtype
        # self.idlist = idlist
        # self.filesuffix = filesuffix
        self._filterargs = kwargs
        self._filtermode=filtermode
        self._readmode=readmode
        self._id_globber = kwargs['idGlobber'] if 'idGlobber' in kwargs else "(^[a-zA-Z0-9]+)"
        self._debug=debugmode
        self._byslices=loadBySlices

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

    def log_print(self, msg, level=logging.INFO):
        logging.getLogger('__main__').log(level, msg)
        if self.verbose:
            print(msg)

    def _parse_root_dir(self):
        """
        Main parsing function.
        """

        if self.verbose:
            print("Parsing root path: ", self.rootdir)

        # Read all nii.gz files exist first.
        removed_fnames = []
        if self._readmode is 'normal':
            file_dirs = os.listdir(self.rootdir)
            file_dirs = fnmatch.filter(file_dirs, "*.nii.gz")
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

        # Apply filter if specified:
        filtered_away = []
        if self._filtermode == 'idlist' or self._filtermode == 'both':
            file_basenames = [os.path.basename(f) for f in file_dirs]
            file_ids = [re.search(self._id_globber, f) for f in file_basenames]
            file_ids = [str(mo.group()) if not mo is None else mo for mo in file_ids]
            if isinstance(self._filterargs['idlist'], str):
                self._idlist = [r.strip() for r in open(self._filterargs['idlist'], 'r').readlines()]
            else:
                self._idlist = self._filterargs['idlist']

            tmp_file_dirs = np.array(file_dirs)
            keep = [id in self._idlist for id in file_ids]
            file_dirs = tmp_file_dirs[keep].tolist()
            filtered_away.extend(tmp_file_dirs[np.invert(keep)])

        if self._filtermode == 'regex' or self._filtermode == 'both':
            file_basenames = [os.path.basename(f) for f in file_dirs]

            # use REGEX if find paranthesis
            if self._filterargs['regex'][0] == '(':
                keep = np.invert([re.match(self._filterargs['regex'], f) is None for f in file_basenames])
                filtered_away.extend(np.array(file_dirs)[np.invert(keep)].tolist())
                file_dirs = np.array(file_dirs)[keep].tolist()
            else: # else use wild card
                file_dirs = fnmatch.filter(file_dirs, "*" + self._filterargs['regex'] + "*")

        if len(removed_fnames) > 0:
            removed_fnames.sort()
            for fs in removed_fnames:
                self.log_print("Cannot find " + fs + " in " + self.rootdir, logging.WARNING)

        file_dirs.sort()

        if self.verbose:
            print("Found %s nii.gz files..."%len(file_dirs))
            print("Start Loading")

        self._itemindexes = [0] # [image index of start slice]
        for i, f in enumerate(tqdm(file_dirs, disable=not self.verbose)) \
                if not self._debug else enumerate(tqdm(file_dirs[:3], disable=not self.verbose)):
            if self.verbose:
                tqdm.write("Reading from "+f)
            im = sitk.ReadImage(f)
            self.dataSourcePath.append(f)
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
        self.length = len(self.dataSourcePath)

        if self._byslices >= 0:
            try:
                self._itemindexes = np.cumsum(self._itemindexes)
                self.length = np.sum([m.size()[self._byslices] for m in self.data])
                # check if all sizes are the same
                allsizes = [tuple(np.array(m.size())[np.arange(m.dim()) != self._byslices]) for m in self.data]
                uniquesizes = set(allsizes)
                if not len(uniquesizes) == 1:
                    logging.log(logging.WARNING, "There are more than one size, attempting to crop")
                    majority_size = uniquesizes[np.argmax([allsizes.count(tup) for tup in uniquesizes])]
                    # Get all index of image that is not of majority size
                    target = [ss != majority_size for ss in allsizes]
                    target = [i for i, x in enumerate(target) if x]

                    for t in target:
                        target_dat = self.data[t]
                        target_size = target_dat.size()
                        cent = np.array(target_size) // 2
                        corner_index = cent - np.array(majority_size)

                        # Crop image to standard size
                        for dim, corn in enumerate(corner_index):
                            t_dim = dim + 1 if dim >= self._byslices else dim
                            temp = target_dat.narrow(t_dim, corn, majority_size[dim])
                        self.data[t] = temp


                self.data = cat(self.data, dim=self._byslices).transpose(0, self._byslices).unsqueeze(1)
            except IndexError:
                print("Wrong Index is used!")
                self.length = len(self.dataSourcePath)
        else:
            try:
                self.data = stack(self.data, dim=0).unsqueeze(1)
            except:
                logging.log(logging.WARNING, "Cannot stack data due to non-uniform shapes.")
                logging.log(logging.INFO, "%s"%[d.shape for d in self.data])
                print("Cannot stack data due to non-uniform shapes. Some function might be impaired.")


    def get_raw_data_shape(self):
        return [self.get_size(id) for id in range(len(self.metadata))]

    def check_shape_identical(self, target_imset):
        assert isinstance(target_imset, ImageDataSet), "Target is not image dataset."

        self_shape = self.get_raw_data_shape()
        target_shape = target_imset.get_raw_data_shape()

        if len(self_shape) != len(target_shape):
            logging.log(logging.WARNING, "Difference data length!")
            return False

        assert type(self_shape) == type(target_shape), "There are major discrepancies in dimension!"
        truth_list = [a == b for a, b in zip(self_shape, target_shape)]
        if not all(truth_list):
            discrip = np.argwhere(np.array(truth_list) == False)
            for x in discrip:
                logging.log(logging.WARNING,
                            "Discripency in element %i, ID: %s, File:[%s, %s]" % (
                                x,
                                self.get_unique_IDs(x),
                                os.path.basename(self.get_data_source(x)),
                                os.path.basename(target_imset.get_data_source(x))))

        return all(truth_list)

    def size(self, int=None):
        if int is None:
            try:
                return self.data.shape
            except:
                return self.length
        else:
            return self.length

    def type(self):
        return self.data[0].type()

    def as_type(self, t):
        try:
            self.data = self.data.type(t)
            self.dtype = t
        except Exception as e:
            print(e)

    def get_data_source(self, i):
        if self._byslices >=0:
            return self.dataSourcePath[int(np.argmax(self._itemindexes > i)) - 1]
        else:
            return self.dataSourcePath[i]

    def get_internal_index(self, i):
        if self._byslices >= 0:
            return i - self._itemindexes[int(np.argmax(self._itemindexes > i)) - 1]
        else:
            return i

    def get_data_by_ID(self, id, globber=None, get_all=False):
        if globber is None:
            globber = self._id_globber

        ids = self.get_unique_IDs(globber)
        if len(set(ids)) != len(ids) and not get_all:
            self.log_print("IDs are not unique using this globber: %s!"%globber, logging.WARNING)

        if ids.count(id) <= 1 or not get_all:
            return self.__getitem__(ids.index(id))
        else:
            return [self.__getitem__(i) for i in np.where(np.array(ids)==id)[0]]


    def get_unique_IDs(self, globber=None):
        """
        Get all IDs globbed by the specified globber. If its None, default globber used. If its not None, the
        class globber will be updated to the specified one.
        """
        import re

        if not globber is None:
            self._id_globber = globber
        filenames = [os.path.basename(self.get_data_source(i)) for i in range(self.__len__())]

        outlist = []
        for f in filenames:
            matchobj = re.search(globber, f)

            if not matchobj is None:
                outlist.append(f[matchobj.start():matchobj.end()])
        return outlist

    def get_size(self, id):
        id = id % len(self.metadata)
        if self._byslices >= 0:
            return [int(self.metadata[self.get_internal_index(id)]['dim[%d]'%(i+1)]) for i in range(3)]
        else:
            return [int(self.metadata[id]['dim[%d]'%(i+1)]) for i in range(3)]

    def get_spacing(self, id):
        id = id % len(self.metadata)
        if self._byslices >= 0:
            return [round(self.metadata[self.get_internal_index(id)]['pixdim[%d]'%(i+1)], 5) for i in range(3)]
        else:
            return [round(self.metadata[id]['pixdim[%d]'%(i+1)], 5) for i in range(3)]

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        from pandas import DataFrame as df
        s = "==========================================================================================\n" \
            "Root Path: %s \n" \
            "Number of loaded images: %i\n" \
            "Load By Slice: %i \n" \
            "Image Details:\n" \
            "--------------\n"%(self.rootdir, self.length, self._byslices)
        # "File Paths\tSize\t\tSpacing\t\tOrigin\n"
        # printable = {'File Name': []}
        printable = {'ID': [], 'File Name': [], 'Size': [], 'Spacing': [], 'Origin': []}
        for i in range(len(self.dataSourcePath)):
            id_mo = re.search(self._id_globber, os.path.basename(self.dataSourcePath[i]))
            id_mo = 'None' if id_mo is None else id_mo.group()
            printable['ID'].append(id_mo)
            printable['File Name'].append(os.path.basename(self.dataSourcePath[i]))
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
        if self._byslices > -1:
            assert self._itemindexes[-1] == tensor_data.size()[0], "Dimension mismatch! (%s vs %s)"%(self._itemindexes[-1], tensor_data.size()[0])
            td=tensor_data.numpy()
            for i in range(len(self.dataSourcePath)):
                start=self._itemindexes[i]
                end=self._itemindexes[i+1]
                # image=sitk.GetImageFromArray(td[start:end])
                templateim = sitk.ReadImage(self.dataSourcePath[i])
                image = sitk.GetImageFromArray(td[start:end])
                image.CopyInformation(templateim)
                # image=self.WrapImageWithMetaData(td[start:end], self.metadata[i])
                sitk.WriteImage(image, outputdirectory+'/'+ prefix + os.path.basename(self.dataSourcePath[i]))

        else:
            assert len(self) == len(tensor_data), "Length mismatch! %i vs %i"%(len(self), len(tensor_data))

            for i in range(len(self)):
                source_file = self.dataSourcePath[i]
                templateim = sitk.ReadImage(source_file)
                image = sitk.GetImageFromArray(tensor_data[i].squeeze().numpy())
                image.CopyInformation(templateim)
                sitk.WriteImage(image, outputdirectory+'/'+ prefix + os.path.basename(self.dataSourcePath[i]))


    @staticmethod
    def WrapImageWithMetaData(inImage, metadata):
        """WrapImageWithMetaData(np.ndarray or sitk.sitkImage) -> sitk.sitkImage

        :param np.ndarray inImage:
        :return:
        """

        im = inImage
        if isinstance(inImage, np.ndarray):
            im = sitk.GetImageFromArray(im)

        if metadata['qform_code'] > 0:
            spacing = np.array([metadata['pixdim[1]'], metadata['pixdim[2]'], metadata['pixdim[3]']],
                               dtype=float)
            ori = np.array([-metadata['qoffset_x'], -metadata['qoffset_y'], metadata['qoffset_z']],
                           dtype=float)
            b = float(metadata['quatern_b'])
            c = float(metadata['quatern_c'])
            d = float(metadata['quatern_d'])
            a = np.sqrt(np.abs(1 - b**2 - c**2 - d**2))
            A = np.array([
                    [a*a + b*b - c*c - d*d, 2*b*c - 2*a*d, 2*b*d + 2*a*c],
                    [2*b*c + 2*a*d , a*a+c*c-b*b-d*d, 2*c*d - 2*a*b],
                    [2*b*d - 2*a*c, 2*c*d + 2*a*b, a*a + d*d - c*c - b*b]
                ])
            A[:2, :3] = -A[:2, :3]
            A[:,2] = float(metadata['pixdim[0]']) * A[:,2]
            im.SetOrigin(ori)
            im.SetDirection(A.flatten())
            im.SetSpacing(spacing)
            return im

    def get_unique_values(self):
        """get_unique_values() -> torch.tensor
        Get the tensor of all unique values in basedata. Only for integer tensors
        :return: torch.tensor
        """
        assert self.data[0].is_floating_point() == False, "This function is for integer tensors. Current datatype is: %s"%(self.data[0].dtype)
        vals = unique(cat([unique(d) for d in self.data]))
        return vals

    def get_unique_values_n_counts(self):
        """get_unique_label_n_counts() -> dict
        Get a dictionary of unique values as key and its counts as value.
        """
        assert self.data[0].is_floating_point() == False, "This function is for integer tensors. Current datatype is: %s"%(self.data[0].dtype)
        out_dict = {}
        for val, counts in [unique(d, return_counts=True) for d in self.data]:
            for v, c in zip(val, counts):
                if v.item() not in out_dict:
                    out_dict[v.item()] = c.item()
                else:
                    out_dict[v.item()] += c.item()
        return out_dict
