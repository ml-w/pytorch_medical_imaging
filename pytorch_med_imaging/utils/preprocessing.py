import SimpleITK as sitk
import os
import numpy as np
import re
import multiprocessing as mpi
from tqdm import *
import random
import argparse
sitk.ProcessObject_GlobalWarningDisplayOff()
from pytorch_med_imaging.logger import Logger

__all__ = ['recursive_list_dir']

def recursive_list_dir(searchDepth, rootdir):
    """
      Recursively lo
    :param searchDepth:
    :param rootdir:
    :return:
    """
    dirs = os.listdir(rootdir)
    nextlayer = []
    for D in dirs:
        if os.path.isdir(rootdir + "/" + D):
            nextlayer.append(rootdir + "/" + D)

    DD = []
    if searchDepth >= 0 and len(nextlayer) != 0:
        for N in nextlayer:
            K = recursive_list_dir(searchDepth - 1, N)
            if not K is None:
                DD.extend(K)

    DD.extend(nextlayer)
    return DD


def SmoothImages(root_dir, out_dir):
    import fnmatch

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    f = os.listdir(root_dir)
    fnmatch.filter(f, "*.nii.gz")

    for fs in f:
        print(fs)
        im = sitk.ReadImage(root_dir + "/" + fs)
        out = sitk.SmoothingRecursiveGaussian(im, 8, True)
        sitk.WriteImage(out, out_dir + "/" + fs)


def make_mask(inimage,
              outdir,
              threshold_lower,
              threshold_upper = None,
              inside_to_1 = True,
              pos=-1):
    r"""Create a mask of an input with specified threshold slice-by-slice.

    Args:
        inimage (str or sitk.Image):
            Input image.
        outdir (str):
            Ouptut directory.
        threshold_lower:
        threshold_upper:
        inside_to_1:
        pos:

    Returns:

    """
    if isinstance(inimage, str):
        print(inimage)
        inimage = sitk.ReadImage(inimage)

    # setup variables
    inside_value = 1 if inside_to_1 else 0
    outside_value = 0 if inside_to_1 else 1

    # might need to cast type correctly in the future

    gttest = sitk.BinaryThreshold(inimage,
                                  upperThreshold=float(threshold_upper),
                                  lowerThreshold=float(threshold_lower),
                                  insideValue=bool(inside_value),
                                  outsideValue=bool(outside_value))
    gttest = sitk.BinaryDilate(gttest, [15, 15, 0], sitk.BinaryMorphologicalOpeningImageFilter.Ball)
    gttest = sitk.BinaryErode(gttest, [15, 15, 0], sitk.BinaryMorphologicalOpeningImageFilter.Ball)
    # gttest = sitk.BinaryMorphologicalClosing(gttest, [0, 25, 25], sitk.BinaryMorphologicalOpeningImageFilter.Ball)
    ss = []

    if pos == -1:
        try:
            pos = int(mpi.current_process().name.split('-')[-1])
        except Exception as e:
            tqdm.write(e)

    try:
        for i in trange(gttest.GetSize()[-1], position=pos, desc=mpi.current_process().name):
            ss.append(sitk.GetArrayFromImage(sitk.BinaryFillhole(gttest[:,:,i])))
        gttest = sitk.GetImageFromArray(np.stack(ss))
        # gttest = sitk.BinaryDilate(gttest, [0, 3, 3], sitk.BinaryDilateImageFilter.Ball)
        gttest.CopyInformation(inimage)
        sitk.WriteImage(gttest, outdir)
        return 0
    except Exception as e:
        print(e)

def make_mask_from_dir(indir, outdir, threshold_lower, threshold_upper, inside_to_1):
    r"""Make mask from a directory"""
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    p = mpi.Pool(10)
    processes = []
    filelist = os.listdir(indir)
    filelist = [indir + '/' + f for f in filelist]

    for i, f in enumerate(filelist):
        outname = f.replace(indir, outdir)
        # make_mask(f, outname, threshold_lower, threshold_upper, inside_to_1)
        subp = p.apply_async(make_mask, (f, outname, threshold_lower, threshold_upper, inside_to_1))
        processes.append(subp)

    for pp in processes:
        pp.wait(50)
    p.close()
    p.join()


