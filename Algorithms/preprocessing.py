import SimpleITK as sitk
import os
import numpy as np
import re
from tqdm import *
sitk.ProcessObject_GlobalWarningDisplayOff()


def RecursiveListDir(searchDepth, rootdir):
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
            K = RecursiveListDir(searchDepth - 1, N)
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
        print fs
        im = sitk.ReadImage(root_dir + "/" + fs)
        out = sitk.SmoothingRecursiveGaussian(im, 8, True)
        sitk.WriteImage(out, out_dir + "/" + fs)


def dicom2nii(folder, out_dir=None):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        assert os.path.isdir(out_dir)

    if not os.path.isdir(folder):
        raise IOError("Cannot locate specified folder!")

    print "Handling: ", folder
    folder = os.path.abspath(folder)
    f = folder.replace('\\', '/')
    # matchobj = re.search('NPC[0-9]+', f)
    matchobj = re.search('[^sS][0-9]{3,4}', f)
    # prefix1 = f.split('/')[-2]
    prefix1 = f[matchobj.start():matchobj.end()]


    # Read file
    series = sitk.ImageSeriesReader_GetGDCMSeriesIDs(f)
    for ss in series:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(
            f,
            ss
        ))
        outimage = reader.Execute()

        # Generate out file name
        headerreader = sitk.ImageFileReader()
        headerreader.SetFileName(reader.GetFileNames()[0])
        headerreader.LoadPrivateTagsOn()
        headerreader.ReadImageInformation()
        outname = out_dir + '/%s-%s.nii.gz'%(prefix1, headerreader.GetMetaData('0008|103e').rstrip().replace(' ', '_'))

        # Write image
        sitk.WriteImage(outimage, outname)
        del reader


def batch_dicom2nii(folderlist, out_dir, workers=8):
    import multiprocessing as mpi
    from functools import partial

    pool = mpi.Pool(workers)
    pool.map_async(partial(dicom2nii, out_dir=out_dir),
                   folderlist)
    pool.close()
    pool.join()


def make_mask(inimage, outdir, pos=0):
    np2sitk = lambda x: sitk.GetImageFromArray(x)
    sitk2np = lambda x: sitk.GetArrayFromImage(x)
    if isinstance(inimage, str):
        inimage = sitk.ReadImage(inimage)

    gttest = sitk.BinaryThreshold(inimage, upperThreshold=65535, lowerThreshold=-400)
    gttest = sitk.BinaryMorphologicalOpening(gttest, [0, 7, 7], sitk.BinaryMorphologicalOpeningImageFilter.Ball)
    ss = []
    for i in trange(gttest.GetSize()[-1], position=pos):
        ss.append(sitk2np(sitk.BinaryFillhole(gttest[:,:,i])))
    gttest = np2sitk(np.stack(ss))

    gttest = sitk.BinaryDilate(gttest, [0, 3, 3], sitk.BinaryDilateImageFilter.Ball)
    gttest.CopyInformation(inimage)

    sitk.WriteImage(gttest, outdir)

def make_mask_from_dir(indir, outdir):
    import multiprocessing as mpi

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    p = mpi.Pool(10)
    processes = []

    filelist = os.listdir(indir)
    filelist = [indir + '/' + f for f in filelist]

    for i, f in enumerate(filelist):
        print i, f
        outname = f.replace(indir, outdir)
        # make_mask(sitk.ReadImage(f), outname, i)
        subp = p.apply_async(make_mask, (f, outname, i))
        processes.append(subp)

    for pp in processes:
        print pp.wait(10)
    p.close()
    p.join()

if __name__ == '__main__':
    # folders = RecursiveListDir(5, '../NPC_Segmentation/00.RAW/Benign NP')
    # batch_dicom2nii(folders, out_dir='../NPC_Segmentation/00.RAW/NIFTI/Benign')
    folders = RecursiveListDir(5, '../NPC_Segmentation/00.RAW/NPC new dx cases')
    batch_dicom2nii(folders, out_dir='../NPC_Segmentation/00.RAW/NIFTI/NPC_dx')