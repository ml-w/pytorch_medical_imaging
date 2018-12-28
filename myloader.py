from MedImgDataset import *
from MedImgDataset.Computation import *
from functools import partial
import os
import numpy as np

# DataLoaders
def LoadSubbandDataset(a, debug=False):
    subband = lambda input, fsuffix, filelist: Subbands(input,
                                                        dtype=np.float32,
                                                        verbose=True,
                                                        debugmode=debug,
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


def LoadImageDataset(a, debug=False):
    image = lambda input, fsuffix, filelist: ImageDataSet(input,
                                                          dtype=np.float32,
                                                          verbose=True,
                                                          debugmode=debug,
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


def LoadSegmentationImageDataset(a, debug=False):
    image = lambda input, fsuffix, filelist, dtype: ImageDataSet(input,
                                                          dtype=dtype,
                                                          verbose=True,
                                                          debugmode=debug,
                                                          filesuffix=fsuffix,
                                                          loadBySlices=0,
                                                          filelist=filelist)

    if a.train is None:
        assert os.path.isfile(a.loadbyfilelist), "Cannot open filelist!"
        # Eval mode
        return image(a.input, a.lsuffix, a.loadbyfilelist, np.float32)
    else:
        # Training Mode
        if a.loadbyfilelist is None:
            return image(a.input, a.lsuffix, None, np.float32), image(a.train, None, None, np.uint8)
        else:
            gt_filelist, input_filelist = a.loadbyfilelist.split(',')
            return image(a.input, a.lsuffix, input_filelist, np.float32), image(a.train, None, gt_filelist, np.uint8)


def LoadSegmentationImageDatasetWithPos(a, debug=False):
    imagewifpos = lambda input, fsuffix, filelist, dtype: ImageDataSetWithPos(input,
                                                          dtype=dtype,
                                                          verbose=True,
                                                          debugmode=debug,
                                                          filesuffix=fsuffix,
                                                          loadBySlices=0,
                                                          filelist=filelist)
    image = lambda input, fsuffix, filelist, dtype: ImageDataSet(input,
                                                          dtype=dtype,
                                                          verbose=True,
                                                          debugmode=debug,
                                                          filesuffix=fsuffix,
                                                          loadBySlices=0,
                                                          filelist=filelist)

    if a.train is None:
        assert os.path.isfile(a.loadbyfilelist), "Cannot open filelist!"
        # Eval mode
        return imagewifpos(a.input, a.lsuffix, a.loadbyfilelist, np.float32)
    else:
        # Training Mode
        if a.loadbyfilelist is None:
            return imagewifpos(a.input, a.lsuffix, None, np.float32), image(a.train, None, None, np.uint8)
        else:
            gt_filelist, input_filelist = a.loadbyfilelist.split(',')
            return imagewifpos(a.input, a.lsuffix, input_filelist, np.float32), image(a.train, None, gt_filelist, np.uint8)


def LoadSegmentationPatchLocTex(a, debug=False):
    imset = lambda input, fsuffix, filelist, dtype: ImageDataSet(input,
                                                          dtype=dtype,
                                                          verbose=True,
                                                          debugmode=debug,
                                                          filesuffix=fsuffix,
                                                          loadBySlices=0,
                                                          filelist=filelist)
    imseg = lambda input, fsuffix, filelist, dtype: ImageDataSet(input,
                                                          dtype=dtype,
                                                          verbose=True,
                                                          debugmode=debug,
                                                          filesuffix=fsuffix,
                                                          loadBySlices=0,
                                                          filelist=filelist)


    patchsize = 128
    stride = 32

    if a.train is None:
        assert os.path.isfile(a.loadbyfilelist), "Cannot open filelist!"
        # Eval mode
        return ImagePatchLocTex(imset(a.input, a.lsuffix, a.loadbyfilelist, np.float32),
                                patchsize, patch_stride=stride * 2, pre_shuffle=True)
    else:
        # Training Mode
        if a.loadbyfilelist is None:
            return ImagePatchLocTex(imset(a.input, a.lsuffix, None, np.float32), patchsize, stride), \
                   ImagePatchesLoader(imseg(a.train, None, None, np.uint8), patchsize, stride)
        else:
            gt_filelist, input_filelist = a.loadbyfilelist.split(',')
            return ImagePatchLocTex(imset(a.input, a.lsuffix, input_filelist, np.float32), patchsize, stride), \
                   ImagePatchesLoader(imseg(a.train, None, gt_filelist, np.uint8), patchsize, stride)


def LoadSegmentationPatchLocTexHist(a, debug=False):
    imset = lambda input, fsuffix, filelist, dtype: ImageDataSet(input,
                                                          dtype=dtype,
                                                          verbose=True,
                                                          debugmode=debug,
                                                          filesuffix=fsuffix,
                                                          loadBySlices=0,
                                                          filelist=filelist)
    imseg = lambda input, fsuffix, filelist, dtype: ImageDataSet(input,
                                                          dtype=dtype,
                                                          verbose=True,
                                                          debugmode=debug,
                                                          filesuffix=fsuffix,
                                                          loadBySlices=0,
                                                          filelist=filelist)


    patchsize = 128
    stride = 32

    if a.train is None:
        assert os.path.isfile(a.loadbyfilelist), "Cannot open filelist!"
        # Eval mode
        return ImagePatchLocTex(imset(a.input, a.lsuffix, a.loadbyfilelist, np.float32),
                                patchsize, patch_stride=stride * 2, pre_shuffle=True, mode='as_histograms')
    else:
        # Training Mode
        if a.loadbyfilelist is None:
            return ImagePatchLocTex(imset(a.input, a.lsuffix, None, np.float32), patchsize, stride, mode='as_histograms'), \
                   ImagePatchesLoader(imseg(a.train, None, None, np.uint8), patchsize, stride)
        else:
            gt_filelist, input_filelist = a.loadbyfilelist.split(',')
            return ImagePatchLocTex(imset(a.input, a.lsuffix, input_filelist, np.float32), patchsize, stride, mode='as_histograms'), \
                   ImagePatchesLoader(imseg(a.train, None, gt_filelist, np.uint8), patchsize, stride)

datamap = {'subband':LoadSubbandDataset,
           'image2D':LoadImageDataset,
           'seg2D': LoadSegmentationImageDataset,
           'seg2DwifPos': LoadSegmentationImageDatasetWithPos,
           'seg2Dloctex': LoadSegmentationPatchLocTex,
           'seg2Dloctexhist': LoadSegmentationPatchLocTexHist,
           'subband_debug': partial(LoadSubbandDataset, debug=True),
           'image2D_debug': partial(LoadImageDataset, debug=True),
           'seg2D_debug': partial(LoadSegmentationImageDataset, debug=True),
           'seg2DwifPos_debug': partial(LoadSegmentationImageDatasetWithPos, debug=True),
           'seg2Dloctex_debug': partial(LoadSegmentationPatchLocTex, debug=True),
           'seg2Dloctexhist_debug': partial(LoadSegmentationPatchLocTexHist, debug=True)
           }