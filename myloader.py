from MedImgDataset import Subbands, ImageDataSet
import os
import numpy as np

# DataLoaders
def LoadSubbandDataset(a):
    subband = lambda input, fsuffix, filelist: Subbands(input,
                                                        dtype=np.float32,
                                                        verbose=True,
                                                        debugmode=False,
                                                        filesuffix=fsuffix,
                                                        loadBySlices=0,
                                                        filelist=filelist
                                                        )
    if a.train is None:
        assert os.path.isfile(a.loadbyfilelist), "Cannot open filelist!"
        # Eval mode
        return subband(a.input, a.lsuffix, a.loadbyfilelist)
    else:
        # Training Mode
        if a.loadbyfilelist is None:
            return subband(a.input, a.lsuffix, None), subband(a.train, None, None)
        else:
            gt_filelist, input_filelist = a.loadbyfilelist.split(',')
            return subband(a.input, a.lsuffix, input_filelist), subband(a.train, None, gt_filelist)


def LoadImageDataset(a):
    image = lambda input, fsuffix, filelist: ImageDataSet(input,
                                                          dtype=np.float32,
                                                          verbose=True,
                                                          debugmode=False,
                                                          filesuffix=fsuffix,
                                                          loadBySlices=0,
                                                          filelist=filelist)

    if a.train is None:
        assert os.path.isfile(a.loadbyfilelist), "Cannot open filelist!"
        # Eval mode
        return image(a.input, a.lsuffix, a.loadbyfilelist)
    else:
        # Training Mode
        if a.loadbyfilelist is None:
            return image(a.input, a.lsuffix, None), image(a.train, None, None)
        else:
            gt_filelist, input_filelist = a.loadbyfilelist.split(',')
            return image(a.input, a.lsuffix, input_filelist), image(a.train, None, gt_filelist)


datamap = {'subband':LoadSubbandDataset,
           'image2D':LoadImageDataset}