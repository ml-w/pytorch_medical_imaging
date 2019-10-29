import os
from functools import partial

import numpy as np

from MedImgDataset import *
from MedImgDataset.Computation import *


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

def LoadSegmentationImageDataset_Aug(a, debug=False):
    image = lambda input, fsuffix, filelist, dtype: ImageDataSet(input,
                                                                 dtype=dtype,
                                                                 verbose=True,
                                                                 debugmode=debug,
                                                                 idlist=filelist,
                                                                 filesuffix=fsuffix,
                                                                 loadBySlices=0
                                                                 )

    imset = lambda input, fsuffix, filelist, dtype: ImageDataSetAugment(input,
                                                                        dtype=dtype,
                                                                        verbose=True,
                                                                        debugmode=debug,
                                                                        filesuffix=fsuffix,
                                                                        loadBySlices=0,
                                                                        idlist=filelist,
                                                                        aug_factor=2)
    imseg = lambda input, fsuffix, filelist, dtype: ImageDataSetAugment(input,
                                                                        dtype=dtype,
                                                                        verbose=True,
                                                                        debugmode=debug,
                                                                        # filesuffix=fsuffix,
                                                                        loadBySlices=0,
                                                                        idlist=filelist,
                                                                        is_seg=True,
                                                                        aug_factor=2)

    if a.train is None:
        assert os.path.isfile(a.loadbyfilelist), "Cannot open filelist!"
        # Eval mode
        return image(a.input, a.lsuffix, a.loadbyfilelist, np.float32)
    else:
        # Training Mode
        invars = imset(a.input, a.lsuffix, a.loadbyfilelist, np.float32)
        seg = imseg(a.train, None, a.loadbyfilelist, np.uint8)
        seg.set_reference_augment_dataset(invars)
        return invars, seg

def LoadSegmentationImageDatasetWithPos(a, debug=False):
    imagewifpos = lambda input, fsuffix, filelist, dtype: ImageDataSet(input,
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
        return ImageDataSetWithPos(imagewifpos(a.input, a.lsuffix, a.loadbyfilelist, np.float32))
    else:
        # Training Mode
        if a.loadbyfilelist is None:
            return ImageDataSetWithPos(imagewifpos(a.input, a.lsuffix, None, np.float32)), \
                   image(a.train, None, None, np.uint8)
        else:
            gt_filelist, input_filelist = a.loadbyfilelist.split(',')
            return ImageDataSetWithPos(imagewifpos(a.input, a.lsuffix, input_filelist, np.float32)), \
                   image(a.train, None, gt_filelist, np.uint8)

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
        # assert os.path.isfile(a.loadbyfilelist), "Cannot open filelist!"
        # Eval mode
        return ImagePatchLocTex(imset(a.input, a.lsuffix, a.loadbyfilelist, np.float32),
                                patchsize, patch_stride=stride * 2, pre_shuffle=True, mode='as_histograms')
    else:
        # Training Mode
        if a.loadbyfilelist is None:
            invars = ImagePatchLocTex(imset(a.input,
                                            a.lsuffix,
                                            None,
                                            np.float32),
                                      patchsize,
                                      stride,
                                      mode='as_histograms',
                                      random_patches=80)
            gtvars = ImagePatchesLoader(imseg(a.train,
                                              None,
                                              None,
                                              np.uint8),
                                        patchsize,
                                        stride,
                                        reference_dataset=invars)
            return invars, gtvars

        else:
            gt_filelist, input_filelist = a.loadbyfilelist.split(',')
            invars = ImagePatchLocTex(imset(a.input,
                                            a.lsuffix,
                                            input_filelist,
                                            np.float32),
                                      patchsize,
                                      stride,
                                      mode='as_histograms',
                                      random_patches=80)
            gtvars = ImagePatchesLoader(imseg(a.train,
                                              None,
                                              gt_filelist,
                                              np.uint8),
                                        patchsize,
                                        stride,
                                        reference_dataset=invars)
            return invars, gtvars

def LoadSegmentationPatchLocTexHist_Aug(a, debug=False):
    imset = lambda input, fsuffix, filelist, dtype: ImageDataSetAugment(input,
                                                                        dtype=dtype,
                                                                        verbose=True,
                                                                        debugmode=debug,
                                                                        filesuffix=fsuffix,
                                                                        loadBySlices=0,
                                                                        idlist=filelist,
                                                                        aug_factor=5)
    imseg = lambda input, fsuffix, filelist, dtype: ImageDataSetAugment(input,
                                                                        dtype=dtype,
                                                                        verbose=True,
                                                                        debugmode=debug,
                                                                        filesuffix=fsuffix,
                                                                        loadBySlices=0,
                                                                        idlist=filelist,
                                                                        is_seg=True)


    patchsize = 128
    stride = 32

    if a.train is None:
        # Eval mode
        imset = lambda input, fsuffix, filelist, dtype: ImageDataSet(input,
                                                                     dtype=dtype,
                                                                     verbose=True,
                                                                     debugmode=debug,
                                                                     filesuffix=fsuffix,
                                                                     loadBySlices=0,
                                                                     filelist=filelist)
        return ImagePatchLocTex(imset(a.input,
                                      a.lsuffix,
                                      a.loadbyfilelist,
                                      np.float32),
                                patchsize, patch_stride=stride, pre_shuffle=True, mode='as_histograms')
    else:
        # Training Mode
        if a.loadbyfilelist is None:
            invars = ImagePatchLocTex(imset(a.input,
                                            a.lsuffix,
                                            None,
                                            np.float32),
                                      patchsize,
                                      stride,
                                      mode='as_histograms',
                                      random_patches=20)
            seg = imseg(a.train, None, None, np.uint8)
            seg.set_reference_augment_dataset(invars._base_dataset)
            gtvars = ImagePatchesLoader(seg,
                                        patchsize,
                                        stride,
                                        reference_dataset=invars)
            return invars, gtvars

        else:
            input_filelist = a.loadbyfilelist
            invars = ImagePatchLocTex(imset(a.input,
                                            a.lsuffix,
                                            input_filelist,
                                            np.float32),
                                      patchsize,
                                      stride,
                                      mode='as_histograms',
                                      random_patches=20)
            seg = imseg(a.train, None, gt_filelist, np.uint8)
            seg.set_reference_augment_dataset(invars._base_dataset)
            gtvarsn = ImagePatchesLoader(seg,
                                        patchsize,
                                        stride,
                                        reference_dataset=invars)
            return invars, gtvars

def LoadSegmentationImageDatasetMMPos_Aug(a, debug=False):
    imset = lambda input, fsuffix, filelist, dtype: ImageDataSetAugment(input,
                                                                        dtype=dtype,
                                                                        verbose=True,
                                                                        debugmode=debug,
                                                                        filesuffix=fsuffix,
                                                                        loadBySlices=0,
                                                                        filelist=filelist,
                                                                        aug_factor=5)

    imseg = lambda input, fsuffix, filelist, dtype: ImageDataSetAugment(input,
                                                                        dtype=dtype,
                                                                        verbose=True,
                                                                        debugmode=debug,
                                                                        filesuffix=fsuffix,
                                                                        loadBySlices=0,
                                                                        filelist=filelist,
                                                                        is_seg=True)
    assert len(a.lsuffix.split(',')) == 2, "Loader suffix should be seperated by commas without spaces."
    suf1, suf2 = a.lsuffix.split(',')
    if a.train is None:
        # Eval mode
        imset = lambda input, fsuffix, filelist, dtype: ImageDataSet(input,
                                                                     dtype=dtype,
                                                                     verbose=True,
                                                                     debugmode=debug,
                                                                     filesuffix=fsuffix,
                                                                     loadBySlices=0,
                                                                     filelist=filelist)

        invars1 = imset(a.input,suf1,a.loadbyfilelist,np.float32)
        invars2 = imset(a.input,suf2,a.loadbyfilelist,np.float32)

        return ImageDataSetWithPos(ImageDataSetMultiChannel(invars1, invars2))
    else:

        # Training Mode
        if a.loadbyfilelist is None:
            invars1 = imset(a.input, suf1, None, np.float32)
            invars2 = imset(a.input, suf2, None, np.float32)
            invars2.set_reference_augment_dataset(invars1)
            invars  = ImageDataSetWithPos(ImageDataSetMultiChannel(invars1, invars2))
            gtvars = imseg(a.train, None, None, np.uint8)
            gtvars.set_reference_augment_dataset(invars1)
            return invars, gtvars
        else:
            gt_filelist, input_filelist = a.loadbyfilelist.split(',')
            invars1 = imset(a.input, suf1, input_filelist, np.float32)
            invars2 = imset(a.input, suf2, input_filelist, np.float32)
            invars2.set_reference_augment_dataset(invars1)
            invars = ImageDataSetWithPos(ImageDataSetMultiChannel(invars1, invars2))
            gtvars = imseg(a.train, None, gt_filelist, np.uint8)
            gtvars.set_reference_augment_dataset(invars1)
            return invars, gtvars

def LoadSegmentationPatchLocMMTexHist_Aug(a, debug=False):
    from MedImgDataset.Computation import clip_5
    imset = lambda input, fsuffix, filelist, dtype: ImageDataSetAugment(input,
                                                                        dtype=dtype,
                                                                        verbose=True,
                                                                        debugmode=debug,
                                                                        filesuffix=fsuffix,
                                                                        loadBySlices=0,
                                                                        idlist=filelist,
                                                                        aug_factor=2)
    imseg = lambda input, fsuffix, filelist, dtype: ImageDataSetAugment(input,
                                                                        dtype=dtype,
                                                                        verbose=True,
                                                                        debugmode=debug,
                                                                        # filesuffix=fsuffix,
                                                                        loadBySlices=0,
                                                                        idlist=filelist,
                                                                        is_seg=True,
                                                                        aug_factor=2)


    patchsize = 128
    stride = 32

    if a.train is None:
        # Eval mode
        imset = lambda input, fsuffix, filelist, dtype: ImageDataSet(input,
                                                                     dtype=dtype,
                                                                     verbose=True,
                                                                     debugmode=debug,
                                                                     filesuffix=fsuffix,
                                                                     loadBySlices=0,
                                                                     idlist=filelist)
        return ImagePatchLocMMTex(imset(a.input,
                                        a.lsuffix,
                                        a.loadbyfilelist,
                                        np.float32),
                                  patchsize,
                                  random_patches=75,
                                  random_from_distribution=clip_5,
                                  renew_index=False,
                                  mode='as_histograms')
    else:
        # Training Mode
        if a.loadbyfilelist is None:
            invars = ImagePatchLocMMTex(imset(a.input,
                                              a.lsuffix,
                                              None,
                                              np.float32),
                                        patchsize,
                                        stride,
                                        mode='as_histograms',
                                        random_patches=15,
                                        random_from_distribution=clip_5
                                        )
            seg = imseg(a.train, None, None, np.uint8)
            seg.set_reference_augment_dataset(invars._base_dataset)
            gtvars = ImagePatchesLoader(seg,
                                        patchsize,
                                        stride,
                                        reference_dataset=invars)
            return invars, gtvars

        else:
            input_filelist = a.loadbyfilelist
            invars = ImagePatchLocMMTex(imset(a.input,
                                              a.lsuffix,
                                              input_filelist,
                                              np.float32),
                                        patchsize,
                                        stride,
                                        mode='as_histograms',
                                        random_patches=15,
                                        random_from_distribution=clip_5
                                        )
            seg = imseg(a.train, None, input_filelist, np.uint8)
            seg.set_reference_augment_dataset(invars._base_dataset)
            gtvars = ImagePatchesLoader(seg,
                                        patchsize,
                                        stride,
                                        reference_dataset=invars)
            return invars, gtvars

def LoadSegmentationPatchRandom_Aug(a, debug=False):
    from MedImgDataset.Computation import clip_5
    imset = lambda input, fsuffix, filelist, dtype: ImageDataSetAugment(input,
                                                                        dtype=dtype,
                                                                        verbose=True,
                                                                        debugmode=debug,
                                                                        filesuffix=fsuffix,
                                                                        loadBySlices=0,
                                                                        filelist=filelist,
                                                                        aug_factor=3)
    imseg = lambda input, fsuffix, filelist, dtype: ImageDataSetAugment(input,
                                                                        dtype=dtype,
                                                                        verbose=True,
                                                                        debugmode=debug,
                                                                        filesuffix=fsuffix,
                                                                        loadBySlices=0,
                                                                        filelist=filelist,
                                                                        is_seg=True,
                                                                        aug_factor=3)


    patchsize = 128
    stride = 32

    if a.train is None:
        # Eval mode
        imset = lambda input, fsuffix, filelist, dtype: ImageDataSet(input,
                                                                     dtype=dtype,
                                                                     verbose=True,
                                                                     debugmode=debug,
                                                                     filesuffix=fsuffix,
                                                                     loadBySlices=0,
                                                                     filelist=filelist)
        return ImagePatchesLoader(imset(a.input,
                                        a.lsuffix,
                                        a.loadbyfilelist,
                                        np.float32),
                                  patchsize,
                                  random_patches=75,
                                  random_from_distribution=clip_5,
                                  renew_index=False)
    else:
        # Training Mode
        if a.loadbyfilelist is None:
            invars = ImagePatchesLoader(imset(a.input,
                                              a.lsuffix,
                                              None,
                                              np.float32),
                                        patchsize,
                                        stride,
                                        random_patches=20,
                                        random_from_distribution=clip_5
                                        )
            seg = imseg(a.train, None, None, np.uint8)
            seg.set_reference_augment_dataset(invars._base_dataset)
            gtvars = ImagePatchesLoader(seg,
                                        patchsize,
                                        stride,
                                        reference_dataset=invars)
            return invars, gtvars

        else:
            gt_filelist, input_filelist = a.loadbyfilelist.split(',')
            invars = ImagePatchesLoader(imset(a.input,
                                              a.lsuffix,
                                              input_filelist,
                                              np.float32),
                                        patchsize,
                                        stride,
                                        random_patches=20,
                                        random_from_distribution=clip_5
                                        )
            seg = imseg(a.train, None, gt_filelist, np.uint8)
            seg.set_reference_augment_dataset(invars._base_dataset)
            gtvars = ImagePatchesLoader(seg,
                                        patchsize,
                                        stride,
                                        reference_dataset=invars)
            return invars, gtvars

def LoadSegmentationImageDatasetByPatches(a, debug=False):
    image = lambda input, fsuffix, filelist, dtype: ImageDataSet(input,
                                                          dtype=dtype,
                                                          verbose=True,
                                                          debugmode=debug,
                                                          filesuffix=fsuffix,
                                                          idlist=filelist)
    image = lambda input, fsuffix, filelist, dtype: ImageDataSet(input,
                                                          dtype=dtype,
                                                          verbose=True,
                                                          debugmode=debug,
                                                          filesuffix=fsuffix,
                                                          idlist=filelist)

    patchsize = [16, 48, 48]
    stride = [16, 48, 48]

    if a.train is None:
        assert os.path.isfile(a.loadbyfilelist), "Cannot open filelist!"
        # Eval mode
        return ImagePatchesLoader3D(image(a.input, a.lsuffix, a.loadbyfilelist, np.float32),
                                    patch_size=patchsize, patch_stride=stride, pre_shuffle=True)
    else:
        # Training Mode
        if a.loadbyfilelist is None:
            imset = ImagePatchesLoader3D(image(a.input, a.lsuffix, None, np.float32),
                                        patch_size=patchsize, patch_stride=stride),
            gtset = ImagePatchesLoader3D(image(a.train, None, None, np.uint8),
                                        patch_size=patchsize, patch_stride=stride, reference_dataset=imset)
            return imset, gtset


        else:
            gt_filelist, input_filelist = a.loadbyfilelist.split(',')
            imset = ImagePatchesLoader3D(image(a.input, a.lsuffix, input_filelist, np.float32),
                                        patch_size=patchsize, patch_stride=stride)
            gtset = ImagePatchesLoader3D(image(a.train, None, gt_filelist, np.uint8),
                                        patch_size=patchsize, patch_stride=stride, reference_dataset=imset)
            return imset, gtset

def LoadClassificationDataSet(a, debug=False):
    image = lambda input, fsuffix, filelist, dtype: ImageDataSet(input,
                                                      dtype=dtype,
                                                      verbose=True,
                                                      debugmode=debug,
                                                      filesuffix=fsuffix,
                                                      idlist=filelist)
    classes = lambda fname: DataLabel.from_csv(fname)

    if a.train is None:
        imset = image(a.input, a.lsuffix, a.loadbyfilelist, np.float32)
        return imset
    else:
        # Training Mode
        imset = image(a.input, a.lsuffix, a.loadbyfilelist, np.float32)
        gtset = classes(a.train)
        gtset.set_target_column('Benign')
        gtset.map_to_data(imset, target_id_globber="(?i)(NPC|P)?[0-9]{3,5}")
        return imset, gtset






datamap = {'subband':LoadSubbandDataset,
           'image2D':LoadImageDataset,
           'seg2D': LoadSegmentationImageDataset,
           'imgclassification': LoadClassificationDataSet,
           'seg2D_aug': LoadSegmentationImageDataset_Aug,
           'seg2Drandompatch_aug': LoadSegmentationPatchRandom_Aug,
           'seg2DwifPos': LoadSegmentationImageDatasetWithPos,
           'seg2DMMwifPos_aug': LoadSegmentationImageDatasetMMPos_Aug,
           'seg2Dloctex': LoadSegmentationPatchLocTex,
           'seg2Dloctexhist': LoadSegmentationPatchLocTexHist,
           'seg2Dloctexhist_aug': LoadSegmentationPatchLocTexHist_Aug,
           'seg2DlocMMtexhist_aug': LoadSegmentationPatchLocMMTexHist_Aug,
           'seg3DPatches': LoadSegmentationImageDatasetByPatches,
           'subband_debug': partial(LoadSubbandDataset, debug=True),
           'image2D_debug': partial(LoadImageDataset, debug=True),
           'imgclassification_debug': partial(LoadClassificationDataSet, debug=True),
           'seg2D_debug': partial(LoadSegmentationImageDataset, debug=True),
           'seg2D_aug_debug': partial(LoadSegmentationImageDataset_Aug, debug=True),
           'seg2Drandompatch_aug_debug': partial(LoadSegmentationPatchRandom_Aug, debug=True),
           'seg2DwifPos_debug': partial(LoadSegmentationImageDatasetWithPos, debug=True),
           'seg2Dloctex_debug': partial(LoadSegmentationPatchLocTex, debug=True),
           'seg2Dloctexhist_debug': partial(LoadSegmentationPatchLocTexHist, debug=True),
           'seg2Dloctexhist_aug_debug': partial(LoadSegmentationPatchLocTexHist_Aug, debug=True),
           'seg2DMMwifPos_aug_debug': partial(LoadSegmentationImageDatasetMMPos_Aug, debug=True),
           'seg2DlocMMtexhist_aug_debug': partial(LoadSegmentationPatchLocMMTexHist_Aug, debug=True),
           'seg3DPatches_debug': partial(LoadSegmentationImageDatasetByPatches, debug=True)
           }
