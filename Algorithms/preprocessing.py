import SimpleITK as sitk
import os
import numpy as np
import re
import sys
import multiprocessing as mpi
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
        print(fs)
        im = sitk.ReadImage(root_dir + "/" + fs)
        out = sitk.SmoothingRecursiveGaussian(im, 8, True)
        sitk.WriteImage(out, out_dir + "/" + fs)


def dicom2nii(folder, out_dir=None, seq_filters=None):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        assert os.path.isdir(out_dir)

    if not os.path.isdir(folder):
        print("Cannot locate specified folder! ", folder)
        raise IOError("Cannot locate specified folder!")

    print("Handling: ", folder)
    folder = os.path.abspath(folder)
    f = folder.replace('\\', '/')
    # matchobj = re.search('NPC[0-9]+', f)
    matchobj = re.search('(?i)(NPC|P)?[0-9]{3,5}', f)
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
        if not seq_filters is None:
            if isinstance(seq_filters, list):
                regex = "("
                for i, fil in enumerate(seq_filters):
                    regex += '(.*' + fil + '{1}.*)'
                    if i != len(seq_filters) - 1:
                        regex += '|'
                regex += ')'
                if re.match(regex, headerreader.GetMetaData('0008|103e')) is None:
                    print("skipping ", headerreader.GetMetaData('0008|103e'), "from ", f)
                    continue
            elif isinstance(seq_filters, str):
                if re.match(seq_filters, headerreader.GetMetaData('0008|103e')) is None:
                    print("skipping ", headerreader.GetMetaData('0008|103e'), "from ", f)
                    continue

        # Write image
        print("Writting: ", outname)
        sitk.WriteImage(outimage, outname)
        del reader


def batch_dicom2nii(folderlist, out_dir, workers=8, seq_fileters=None):
    import multiprocessing as mpi
    from functools import partial

    pool = mpi.Pool(workers)
    # pool.map_async(partial(dicom2nii, out_dir=out_dir, seq_fileters=seq_fileters),
    #                folderlist)
    for f in folderlist:
        p = pool.apply_async(dicom2nii, args=[f, out_dir, seq_fileters])
    pool.close()
    pool.join()


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

    folders = RecursiveListDir(5, args[1])
    folders = [os.path.abspath(f) for f in folders]

    if isinstance(eval(args[3]), list):
        batch_dicom2nii(folders, args[2], eval(args[3]) if len(args) > 4 else None)
    else:
        batch_dicom2nii(folders, args[2], args[3] if len(args) > 4 else None)

if __name__ == '__main__':
    # folders = RecursiveListDir(5, '../NPC_Segmentation/00.RAW/Benign NP')
    # batch_dicom2nii(folders, out_dir='../NPC_Segmentation/00.RAW/NIFTI/Benign')
    # folders = RecursiveListDir(5, '../NPC_Segmentation/00.RAW/T1+C_Missing/t1c/')
    # folders = RecursiveListDir(5, '../NPC_Segmentation/00.RAW/MMX/840/')
    # batch_dicom2nii(folders, out_dir='../NPC_Segmentation/00.RAW/NIFTI/All')
    # dicom2nii('../NPC_Segmentation/00.RAW/MMX/769/S', '../NPC_Segmentation/00.RAW/NIFTI/MMX')
    batch_dicom2nii(RecursiveListDir(3, '../NPC_Segmentation/00.RAW/Transfer/Malignant2'),
                    '../NPC_Segmentation/0A.NIFTI_ALL/Malignant')
    # dicom2nii('../NPC_Segmentation/00.RAW/Transfer/Benign/NPC147/Orignial Scan/DICOM', '../NPC_Segmentation/0A.NIFTI_ALL/Benign')
    # main(sys.argv)
    # make_mask_from_dir('../NPC_Segmentation/06.NPC_Perfect/temp_t2/', '../NPC_Segmentation/06.NPC_Perfect/temp_mask')