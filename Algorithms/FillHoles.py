import SimpleITK as sitk
import numpy as np
import os
import fnmatch
import visdom
import multiprocessing as mpi
from MedImgDataset import ImageDataSet
from tqdm import tqdm

__all__ = ['ApplyMask']

vis = visdom.Visdom(server="http://223.255.146.2", port=8097)

def ParseRootdir(dir):
    """
    Description
    -----------
      Process the root dir and identify unique sample

    :param str dir
    :return:
    """

    filenames = fnmatch.filter(os.listdir(dir), "*.npy")
    recon_projection_numbers = [name.split('_')[1].split('.')[0] for name in filenames]
    filenames = [name.split('_')[0] for name in filenames]
    filenames = list(set(filenames))
    assert len(filenames) > 0, "No npy files were found in root directory: %s"%dir

    return filenames

def FillHole2D(array, mask=True):
    """
    Description
    -----------
      Fill holes in 2D space instead of 3D so holes connected to the image boundary
      will still be filled.

    :param np.ndarray array: 2D array to be filled, assume (H, W)
    :param bool mask:        Use circular mask or not
    :return:
    """
    assert isinstance(array, np.ndarray), "Input has to be array"
    assert array.ndim == 2, "Input has to be 2D"

    m = None
    if mask:
         s = array.shape
         r = float(s[1] / s[0])
         x, y, = np.meshgrid(np.arange(0, s[0]), np.arange(0, s[1]))
         m = (x - s[0]/2)**2. + (y - s[0]/2)**2 / r > (s[0] / 2.)**2 + 1

    array[m] = -3024
    dis = np.array(array, dtype=float)
    dis -= dis.min()
    dis /= dis.max()
    # vis.image(dis, win="Before")

    im = sitk.GetImageFromArray(array)
    im = sitk.BinaryThreshold(im,  lowerThreshold=-400, upperThreshold=5000, insideValue=1)
    im = sitk.BinaryFillhole(im)
    im = sitk.BinaryErode(im, 9)
    im = sitk.BinaryDilate(im, 12)

    im = np.array(sitk.GetArrayFromImage(im), dtype=float)
    im[m] = 0

    # vis.image(im, env="main", win="FillHole")
    return im

def FillHole3D(array, mask=True):
    """
    Description
    -----------
      Assume input array is in shape of [B, H, W]

    :param np.ndarray array: 3D array to be filled
    :param bool mask:        Use circular mask or not
    :return:
    """
    assert isinstance(array, np.ndarray), "Input has to be array"
    assert array.ndim == 3, "Input has to be 3D"

    im = []
    for i in xrange(array.shape[0]):
        l_im = FillHole2D(array[i], mask)
        im.append(l_im)

    im = [img.reshape(1, img.shape[0], img.shape[1]) for img in im]
    return np.concatenate(im, 0)

def SaveImage(array, prefix):
    """
    Description
    -----------
      Save the numpy image assuming dimension is (B, H, W) along the batch (i.e. slice)
      direction. Each slices are saved as separated .npy files with the format
      @prefix + "_msk_S%03d"%slicenum

    :param np.ndarray array: Array to be saved, assume (B, H, W)
    :param str prefix:       Prefix of the saved files
    :return:
    """


    assert isinstance(array, np.ndarray)
    assert isinstance(prefix, str)
    assert array.ndim == 3, "Only support 3D images (B, H, W)"

    for i in xrange(array.shape[0]):
        np.save(prefix + "_msk_S%03d"%i, array[i])

def ProcessDirectory(dir, outdir):
    """
    Description
    -----------
      Process all the npy files with fill hole methods.

    :param str dir:
    :return:
    """

    assert os.path.isdir(dir), "Directory doesn't exist"

    fns = ImageDataSet(dir, verbose=True)

    for i in tqdm(xrange(len(fns))):
        fn = fns.dataSourcePath[i].replace(dir, outdir)
        im = fns[i].numpy()
        im = FillHole3D(im)
        im = np.array(im, dtype=np.uint8)
        sitk.WriteImage(sitk.GetImageFromArray(im), fn)


def ShowMaskOnVisdom(dir):
    """
    Description
    -----------
      Show the result in visdom server (env = "main")

    :param dir:
    :return:
    """
    assert os.path.isdir(dir), "Directory doesn't exist"

    fns = ParseRootdir(dir)
    assert len(fns) != 0, "Directory is empty"

    for i in xrange(len(fns)):
        print "Working on ", i
        fs = os.listdir(dir)
        f1 = fnmatch.filter(fs, fns[i] + "*_msk_*")
        f2 = fnmatch.filter(fs, fns[i] + "*_128_*")
        f1.sort()
        f2.sort()


        im1 = [np.load(dir + "/" + f) for f in f1]
        im1 = [np.array(im, dtype=float) for im in im1]
        im1 = [im - im.min() for im in im1]
        im1 = [im / im.max() for im in im1]
        im2 = [np.load(dir + "/" + f) for f in f2]
        im2 = [np.array(im, float) for im in im2]
        im2 = [im - im.min() for im in im2]
        im2 = [im / im.max() for im in im2]

        for j in xrange(len(im1)):
            vis.image(im1[j], win="Im1")
            vis.image(im2[j], win="Im2")


def ApplyMask(im, mask, outvalue=-3024, inverse=False, output=None):
    """
    Description
    -----------
      Apply binary mask on image. Assume input to be ITK images

    :param SimpleITK.Image im:   Input
    :param SimpleITK.Image mask: Mask of input
    :param float outvalue:
    :param bool inverse
    :return:
    """

    assert isinstance(inverse, bool), "Error in arguments!"

    if (isinstance(im, str)):
        print "Reading!"
        im = sitk.ReadImage(im)
    if (isinstance(mask, str)):
        mask = sitk.ReadImage(mask)

    assert im.GetDimension() == mask.GetDimension(), "Image and mask must have same dimension!"
    if (mask.GetPixelID() != sitk.sitkUInt8):
        mask = sitk.Cast(mask, sitk.sitkUInt8)

    maskfilter = sitk.MaskImageFilter()
    maskfilter.SetOutsideValue(outvalue)
    outputimage = maskfilter.Execute(im, mask)

    if not(output is None):
        print "Saving to ", output
        sitk.WriteImage(outputimage, output)

    del im, mask
    return outputimage

def BatchApplyMask(imlist, masklist, outvalue=-3024, inverse=False):
    """
    Description
    -----------
      Apply mask using the function ApplyMask.

    :param imlist:
    :param masklist:
    :param outvalue:
    :param inverse:
    :return:
    """

    assert len(imlist) == len(masklist), "Image list and mask list has different length!"

    pool = mpi.Pool(processes=5)
    p = []
    for i in xrange(len(imlist)):
        imfn = imlist[i]
        mkfn = masklist[i]

        # Skip all masked images
        if imfn.find('m.nii.gz') != -1:
            print "Skipping ", imfn
            continue

        outfn = imfn.replace('.nii.gz', 'm.nii.gz')
        if not(os.path.isfile(imfn) and os.path.isfile(mkfn)):
            print "Path doesn't exist: ", imfn, mkfn
            continue

        # im = sitk.ReadImage(imfn)
        # mk = sitk.ReadImage(mkfn)
        process = pool.apply_async(ApplyMask, (imfn, mkfn, outvalue, inverse, outfn))
        print "Creating job: ", i
        p.append(process)

    for process in p:
        process.wait()

    pass

def MaskDirectory(target, mask):
    """
    Description
    -----------
      Use Batch Apply Mask to process a directory

    :param target:
    :param mask:
    :return:
    """

    assert len(target) == len(mask), "Two directory lists must have the same length!"

    tlist = []
    mlist = []

    for i in xrange(len(target)):
        if not(os.path.isdir(target[i]) and os.path.isdir(mask[i])):
            print "Cannot open directory pair: ", target[i], mask[i]
            continue

        t = os.listdir(target[i])
        m = os.listdir(mask[i])

        t = fnmatch.filter(t, "*.nii.gz")
        t.sort()
        for f in t:
            if f.find('m.nii.gz') != -1:
                continue

            if f.split('_')[0] + "_msk.nii.gz" in m:
                tlist.append(target[i] + "/" + f)
                mlist.append(mask[i] + "/" + f.split('_')[0] + "_msk.nii.gz")

    BatchApplyMask(tlist, mlist, outvalue=-1000)

def MaskedMerge(im1, im2, mask):
    """
    Description
    -----------
      Merge two images using a mask. The product will be the addition of
      im1*mask and image2 * (-mask).

      Assume all inputs are simple itk images.

    :param im1:
    :param im2:
    :param mask:
    :return:
    """

    assert isinstance(im1, sitk.Image) and isinstance(im2, sitk.Image), \
        "Input image must be sitk images!"
    assert isinstance(mask, sitk.Image), "Input mask must be sitk image!"

    if (mask.GetPixelID() != sitk.sitkUInt8):
        mask = sitk.Cast(mask, sitk.sitkUInt8)

    maskfilter1 = sitk.MaskImageFilter()
    masked1 = maskfilter1.Execute(im1, mask)

    invertmask = sitk.InvertIntensity(mask)
    maskfilter2 = sitk.MaskImageFilter()
    masked2 = maskfilter2.Execute(im2, invertmask)

    addfilter = sitk.AddImageFilter()
    output = addfilter.Execute(masked1, masked2)

    return output

def main():
    ProcessDirectory("../LCTSC/99.TempTrain/Inputs/32", "../LCTSC/99.TempTrain/Masks/")

if __name__ == '__main__':
    main()


