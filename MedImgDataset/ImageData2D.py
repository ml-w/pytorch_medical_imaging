from torch.utils.data import Dataset
from torch import from_numpy, cat
import fnmatch
import os
import numpy as np
from skimage.io import imread
import imgaug as ia
import imgaug.augmenters as iaa
from tqdm import tqdm


class ImageDataSet2D(Dataset):
    def __init__(self, rootdir, as_grey=True, recursive=False, verbose=False, dtype=float, readfunc=None, resizetosquare=-1):
        """ImageDataSet2D
        Description
        -----------
          This class read 2D images with png/jpg file extension in the specified folder into torch tensor.

        :param str rootdir:  Specify which directory to look into
        :param str readmode: This argument is passed to skimage.io.imread
        :param bool recursive: If this is true, walk through all subdirectories
        :param bool verbose: Set to True if you want verbose info
        :param callable readfunc: If this is set, it will be used to load image files, as_grey option will be ignored
        :param type dtype:   The type to cast the tensors
        """
        super(ImageDataSet2D, self).__init__()
        assert os.path.isdir(rootdir), "Cannot access directory!"
        self.rootdir = rootdir
        self.dataSourcePath = []
        self.data = []
        self.length = 0
        self.verbose = verbose
        self.dtype = dtype
        self.as_grey=as_grey
        self.readfunc = readfunc
        self._recursive = recursive
        self._metadata = {'Original Size': []}
        self._ParseRootDir()
        self._resize = resizetosquare

    def _ParseRootDir(self):
        """
        Description
        -----------
          Load all .png, .jpg images to cache
        :return:
        """

        if not self._recursive:
            filenames = os.listdir(self.rootdir)
            filenames.sort()
            [self.dataSourcePath.extend(fnmatch.filter(filenames, "*" + ext)) for ext in ['.png','.jpg', '.JPG', '.PNG']]
            self.dataSourcePath = [self.rootdir + "/" + F for F in self.dataSourcePath]
            self.dataSourcePath.sort()
        else:
            filenames = []
            for root, dirnames, fnames in os.walk(self.rootdir):
                for filename in fnames:
                    if filename.endswith(('.jpg', '.png', '.PNG', '.JPG')):
                        filenames.append(os.path.join(root, filename))
            filenames.sort()
            self.dataSourcePath = filenames

        for f in tqdm(self.dataSourcePath, disable=not self.verbose):
            if self.verbose:
                tqdm.write("Reading from "+f)
            if self.readfunc is None:
                im = imread(f, as_grey=self.as_grey)
            else:
                im = self.readfunc(f)

            self._metadata['Original Size'].append(im.shape)
            im = from_numpy(np.array(im*255, dtype=self.dtype)) if self.as_grey and self.dtype == np.uint8 else \
                from_numpy(np.array(im, dtype=self.dtype))
            self.data.append(im)

        self.length = len(self.data)


    def __getitem__(self, item):
        if self._resize < 0:
            return self.data[item]
        else:
            return self.ResizeToSquare(self.data[item])

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
        printable['Original Size'] = self._metadata['Original Size']

        data = df(data=printable)
        s += data.to_string()
        return s

    def __len__(self):
        return self.length

    def tonumpy(self):
        assert self.length != 0
        return cat([K.unsqueeze(0) for K in self.data], dim=0).numpy()

    def ResizeToSquare(self, imagedata):
        # Get data
        im = (imagedata.numpy() * 255.).astype('uint8')

        b = im.shape[0] > im.shape[1]  # True if height > width
        s = 512 / float(max(im.shape))
        py, px = np.round(self._resize - np.array(im.shape).astype('float32') * s)
        pl, pr = int(np.floor(px / 2.)), int(np.ceil(px/2.))
        pu, pd = int(np.floor(py / 2.)), int(np.ceil(py/2.))

        # seq = iaa.Sequential([iaa.Pad(px=(pu, pr, pd, pl)),
        seq = iaa.Sequential([iaa.Scale({'height':self._resize, "width": "keep-aspect-ratio" } if b else
                                        {'width':self._resize, "height": "keep-aspect-ratio" } ),
                              iaa.Pad(px=(pu, pr, pd, pl), pad_mode='constant', pad_cval=0, keep_size=False),
                              iaa.Scale({'height':self._resize, "width":self._resize})]) # top, right, bottom, left
        seq_det = seq.to_deterministic()
        im_aug = seq_det.augment_image(im)
        return from_numpy(im_aug.astype(self.dtype)/255.)


