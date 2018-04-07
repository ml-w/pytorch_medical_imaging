from torch.utils.data import Dataset
from torch import from_numpy
from tqdm import tqdm
import fnmatch
import os
import numpy as np
import pydicom
import SimpleITK as sitk
import gc


class Projection(Dataset):
    def __init__(self, rootdir, verbose=False, dtype=np.float32, cachesize=8):
        """ImageDataSet2D
        Description
        -----------
          This class read 2D images with png/jpg file extension in the specified folder into torch tensor.

        :param str rootdir:  Specify which directory to look into
        :param bool verbose: Set to True if you want verbose info
        :param callable readfunc: If this is set, it will be used to load image files, as_grey option will be ignored
        :param type dtype:   The type to cast the tensors
        """
        super(Projection, self).__init__()
        assert os.path.isdir(rootdir), "Cannot access directory!"
        self.rootdir = rootdir
        self.dataSourcePath = []
        self.datacache = {}
        self.datasize = []
        self.datamemsize = 0
        self.length = 0
        self.verbose = verbose
        self.dtype = dtype
        self.cachesize = cachesize
        self._ParseRootDir()

    def _ParseRootDir(self):
        """
        Description
        -----------

        :return:
        """

        filenames = os.listdir(self.rootdir)
        filenames.sort()
        self.dataSourcePath = [self.rootdir + "/" + F for F in filenames]
        self.dataSourcePath.sort()
        self.length = len(self.dataSourcePath)

        temp = pydicom.dcmread(self.dataSourcePath[0]).pixel_array
        self.datasize = [self.length]
        self.datasize.extend(temp.shape)
        self.datamemsize = temp.nbytes
        del temp

    def __getitem__(self, item):
        try:
            return from_numpy(self.datacache[item].astype('float'))
        except KeyError:
            try:
                k = pydicom.dcmread(self.dataSourcePath[item], force=True)

                # Array has to be rescaled according to mayo clinic's data structure
                C = k.RescaleIntercept
                M = k.RescaleSlope
                arr = np.array(k.pixel_array, dtype=np.float32) * M + C

                # delete some items if the cahche size exceeds a certain limit
                while self._CalChacheSize() > self.cachesize * (1024 ** 3) / 8.:
                    dictitem = self.datacache.popitem()
                    del dictitem
                    gc.collect()

                self.datacache[item] =arr.astype(self.dtype)
                return self[item]
            except AttributeError:
                tqdm.write("File %s has some missing attributes! Returning random image instead!"%self.dataSourcePath[item])
                return from_numpy(self.datacache[self.datacache.keys()[0]])
            except:
                tqdm.write("Unknow error at file %s"%self.dataSourcePath[item])
                return from_numpy(self.datacache[self.datacache.keys()[0]])


    def __str__(self):
        from pandas import DataFrame as df
        s = "==========================================================================================\n" \
            "Root Path: %s \n" \
            "Number of loaded images: %i\n" \
            "Image Details:\n" \
            "--------------\n"%(self.rootdir, self.length)
        # "File Paths\tSize\t\tSpacing\t\tOrigin\n"
        # printable = {'File Name': []}
        printable = {'File Name': [], 'Size': []}
        for i in xrange(self.length):
            printable['File Name'].append(os.path.basename(self.dataSourcePath[i]))
            # for keys in self.metadata[i]:
            #     if not printable.has_key(keys):
            #         printable[keys] = []
            #
            #     printable[keys].append(self.metadata[i][keys])
            printable['Size'].append([self.__getitem__(i).size()[0],
                                      self.__getitem__(i).size()[1]])

        data = df(data=printable)
        s += data.to_string()
        return s

    def __len__(self):
        return self.length

    def _CalChacheSize(self):
        """_CalChaceSize(self) -> int

        :return: size in bytes
        """
        return len(self.datacache) * self.datamemsize

    def size(self, int):
        return self.datasize[int]


    def Write(self, tensor, out_rootdir):
        assert tensor.size()[0] == self.__len__(), "Length is in correct!"

        tensor = tensor.numpy()

        for i in tqdm(range(tensor.shape[0])):
            k = pydicom.dcmread(self.dataSourcePath[i], force=True)
            tosave = (tensor[i] - k.RescaleIntercept) / k.RescaleSlope
            tosave = tosave.astype(np.uint16)
            k.PixelData = tosave.tostring()
            k.save_as(self.dataSourcePath[i].replace(self.rootdir, out_rootdir + "/"))