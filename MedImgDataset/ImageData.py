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
    """
    This dataset automatically load all the nii files in the specific directory to
    generate a 3D dataset
    """
    def __init__(self, rootdir, filelist=None, filesuffix=None, idlist=None, loadBySlices=-1, verbose=False, dtype=float, debugmode=False):
        """

        :param rootdir:
        """
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
        self.filelist = filelist
        self.idlist = idlist
        self.filesuffix = filesuffix
        self._debug=debugmode
        self._byslices=loadBySlices
        self._ParseRootDir()


    def _ParseRootDir(self):
        """
        Description
        -----------
          Load all nii images to cache

        :return:
        """

        if self.verbose:
            print("Parsing root path: ", self.rootdir)

        # Load files written in filelist from the root_dir
        removed_fnames = []
        if not self.filelist is None:
            filelist = open(self.filelist, 'r')
            filenames = [fs.rstrip() for fs in filelist.readlines()]
            for fs in filenames:
                if not os.path.isfile(self.rootdir + '/' + fs):
                    filenames.remove(fs)
                    removed_fnames.append(fs)
        elif not self.idlist is None:
            if isinstance(self.idlist, str):
                self.idlist = [r.strip() for r in file(self.idlist, 'r').readlines()]
            tmp_filenames = os.listdir(self.rootdir)
            tmp_filenames = fnmatch.filter(tmp_filenames, "*.nii.gz")
            filenames = []
            for id in self.idlist:
                filenames.extend(fnmatch.filter(tmp_filenames, str(id)+"*"))
        else:
            filenames = os.listdir(self.rootdir)
            filenames = fnmatch.filter(filenames, "*.nii.gz")

        if not self.filesuffix is None:
            # use REGEX if find paranthesis
            if self.filesuffix[0] == '(':
                tmp = []
                for f in filenames:
                    if not re.match(self.filesuffix, f) is None:
                        tmp.append(f)
                filenames = tmp
            else:
                # Else use bash-style wild card
                filenames = fnmatch.filter(filenames, "*" + self.filesuffix + "*")

        if len(removed_fnames) > 0:
            removed_fnames.sort()
            for fs in removed_fnames:
                logging.getLogger('__main__').log(logging.WARNING, "Cannot find " + fs + " in " + self.rootdir)
                print("Cannot find " + fs + " in " + self.rootdir)
        filenames.sort()

        if self.verbose:
            print("Found %s nii.gz files..."%len(filenames))
            print("Start Loading")

        self._itemindexes = [0] # [image index of start slice]
        for i, f in enumerate(tqdm(filenames, disable=not self.verbose)) \
                if not self._debug else enumerate(tqdm(filenames[:3], disable=not self.verbose)):
            if self.verbose:
                tqdm.write("Reading from "+f)
            im = sitk.ReadImage(self.rootdir + "/" + f)
            self.dataSourcePath.append(self.rootdir + "/" + f)
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

    def get_unique_IDs(self, globber=None):
        import re
        if globber is None:
            globber = "[^T][0-9]+"

        filenames = [os.path.basename(f) for f in self.dataSourcePath]


        outlist = []
        for f in filenames:
            matchobj = re.search(globber, f)

            if not matchobj is None:
                outlist.append(int(f[matchobj.start():matchobj.end()]))
        return outlist

    def get_spacing(self, id):
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
        printable = {'File Name': [], 'Size': [], 'Spacing': [], 'Origin': []}
        for i in range(len(self.dataSourcePath)):
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
        s += data.to_string()
        return s

    def Write(self, tensor_data, outputdirectory, prefix=''):
        if self._byslices > -1:
            assert self._itemindexes[-1] == tensor_data.size()[0], "Dimension mismatch!"
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
            A[:,2] = metadata['pixdim[0]'] * A[:,2]
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
