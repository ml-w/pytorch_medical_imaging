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


def make_mask(inimage, outdir, pos=-1):
    if isinstance(inimage, str):
        # tqdm.write("Reading " + inimage)
        inimage = sitk.ReadImage(inimage)

    gttest = sitk.BinaryThreshold(inimage, upperThreshold=65535, lowerThreshold=200)
    gttest = sitk.BinaryDilate(gttest, [15, 15, 0], sitk.BinaryMorphologicalOpeningImageFilter.Ball)
    gttest = sitk.BinaryErode(gttest, [15, 15, 0], sitk.BinaryMorphologicalOpeningImageFilter.Ball)
    # gttest = sitk.BinaryMorphologicalClosing(gttest, [0, 25, 25], sitk.BinaryMorphologicalOpeningImageFilter.Ball)
    ss = []

    if pos == -1:
        try:
            pos = int(mpi.current_process().name.split('-')[-1])
        except Exception as e:
            tqdm.write(e.message)

    try:
        for i in trange(gttest.GetSize()[-1], position=pos, desc=mpi.current_process().name):
            ss.append(sitk.GetArrayFromImage(sitk.BinaryFillhole(gttest[:,:,i])))
        gttest = sitk.GetImageFromArray(np.stack(ss))
        # gttest = sitk.BinaryDilate(gttest, [0, 3, 3], sitk.BinaryDilateImageFilter.Ball)
        gttest.CopyInformation(inimage)
        sitk.WriteImage(gttest, outdir)
        return 0
    except Exception as e:
        print(e.message)

def make_mask_from_dir(indir, outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    p = mpi.Pool(10)
    processes = []
    filelist = os.listdir(indir)
    filelist = [indir + '/' + f for f in filelist]

    for i, f in enumerate(filelist):
        outname = f.replace(indir, outdir)
        # make_mask(sitk.ReadImage(f), outname, 0)
        subp = p.apply_async(make_mask, (f, outname, -1))
        processes.append(subp)

    for pp in processes:
        pp.wait(50)
    p.close()
    p.join()


def main(args):
    try:
        os.makedirs(args[2], exist_ok=True)
    except:
        print("Cannot mkdir.")

    assert os.path.isdir(args[1]) and os.path.isdir(args[2]), 'Cannot locate inputs directories or output directory.'

    folders = recursive_list_dir(5, args[1])
    folders = [os.path.abspath(f) for f in folders]

    if isinstance(eval(args[3]), list):
        batch_dicom2nii(folders, args[2], eval(args[3]) if len(args) > 4 else None)
    else:
        batch_dicom2nii(folders, args[2], args[3] if len(args) > 4 else None)
