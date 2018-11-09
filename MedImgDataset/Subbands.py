from torch.utils.data import Dataset
from torch import from_numpy, cat
from tqdm import tqdm
import fnmatch
import os
import numpy as np
import gc

class Subbands(Dataset):
    def __init__(self, rootdir, filelist = None, loadBySlices=-1, filesuffix=None, verbose=False, dtype=np.float32, debugmode=False):
        """Subbands
        Description
        -----------


        :param str rootdir:  Specify which directory to look into
        :param bool verbose: Set to True if you want verbose info
        :param callable readfunc: If this is set, it will be used to load image files, as_grey option will be ignored
        :param type dtype:   The type to cast the tensors
        """
        super(Subbands, self).__init__()
        assert os.path.isdir(rootdir), "Cannot access directory!"
        self.rootdir = rootdir
        self.dataSourcePath = []
        self.data = []
        self.length = 0
        self.verbose = verbose
        self.dtype = dtype
        self.filesuffix = filesuffix
        self.filelist = filelist
        self._debug=debugmode
        self._byslices=loadBySlices
        self._ParseRootDir()

    def _ParseRootDir(self):
        """
        Description
        -----------

        :return:
        """

        # Load files written in filelist from the root_dir
        if self.filelist is None:
            filenames = os.listdir(self.rootdir)
            filenames = fnmatch.filter(filenames, "*.npz")
        else:
            filelist = open(self.filelist, 'r')
            filenames = [fs.rstrip() for fs in filelist.readlines()]
            for fs in filenames:
                if not os.path.isfile(self.rootdir + '/' + fs):
                    filenames.remove(fs)
                    print "Cannot find " + fs + " in " + self.rootdir

        if not self.filesuffix is None:
            filenames = fnmatch.filter(filenames, "*" + self.filesuffix + "*")
        filenames.sort()

        self.length = len(filenames)
        if self.verbose:
            print "Found %s subband files..."%self.length
            print "Start Loading"

        self._itemindexes = [0] # [image index of start slice]
        for i, f in enumerate(tqdm(filenames, disable=not self.verbose)) \
                if not self._debug \
                else enumerate(tqdm(filenames[:3], disable=not self.verbose)):
            if self.verbose:
                tqdm.write("Reading from "+f)
            im = np.load(self.rootdir + '/' + f)['subbands']
            self.dataSourcePath.append(self.rootdir + "/" + f)
            self.data.append(from_numpy(im.transpose(0, 3, 1, 2).astype(self.dtype))) # align with B x C x H x W
            self._itemindexes.append(self.data[i].size()[0])

        if self._byslices >= 0:
            try:
                self._itemindexes = np.cumsum(self._itemindexes)
                self.dataSourcePath = [p for k in xrange(len(self.dataSourcePath))
                                         for p in [self.dataSourcePath[k]]*self.data[k].size()[self._byslices]]
                self.length = np.sum([m.size()[self._byslices] for m in self.data])
                self.data = cat(self.data, dim=self._byslices).transpose(0, self._byslices)
            except IndexError:
                print "Wrong Index is used!"
                self.length = len(self.dataSourcePath)


    def __getitem__(self, item):
        return self.data[item]

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
        for i in xrange(len(self.data)):
            printable['File Name'].append(os.path.basename(self.dataSourcePath[i]))
            printable['Size'].append([j for j in self.__getitem__(i).size()])

        data = df(data=printable)
        s += data.to_string()
        return s

    def __len__(self):
        return self.length

    def size(self, i):
        return self.data.size(i)

    def Write(self, tensor, out_rootdir):
        assert tensor.size()[0] == self.__len__(), "Length is incorrect!"
        tensor = tensor.numpy().transpose(0, 2, 3, 1)

        for i in tqdm(range(len(self._itemindexes)-1)):
            outfname = self.dataSourcePath[self._itemindexes[i]].replace(self.rootdir, out_rootdir)
            np.savez_compressed(outfname, subbands=tensor[self._itemindexes[i]:
                                                          self._itemindexes[i+1]])
            pass
