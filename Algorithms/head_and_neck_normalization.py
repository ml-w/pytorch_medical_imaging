from eros import *
from tqdm import *
import SimpleITK as sitk
import numpy as np
import sys, os
from utils import *
from crop_image import *
from NyulNormalizer import nyul
from shutil import *

def transform_label(label_im, transform):
    """
    Apply transform on label image.
    """
    assert isinstance(label_im, sitk.Image)
    assert isinstance(transform, sitk.AffineTransform)

    tempim = sitk.GetImageFromArray(sitk.GetArrayFromImage(label_im))

    out_im = sitk.Resample(tempim, transform)
    out_im.CopyInformation(label_im)
    return out_im


def align_image_to_symmetry_plane(image):
    """
    Rotate input image to align with plane of coronal symmetry
    """
    assert isinstance(image, sitk.Image)

    ssfactor    = 4

    eros_res    = eros.eros(sitk.GetArrayFromImage(image)[:,::ssfactor,::ssfactor], 2, angle_range=[-10, 10])
    best_angle  = eros_res.get_mean_angle()
    com         = eros_res.get_mean_com() * ssfactor

    # strip directional information
    newim = sitk.GetImageFromArray(sitk.GetArrayFromImage(image))
    s = np.array(newim.GetSize())

    # perform shift
    v =  np.array(com) - (np.array(image.GetSize()[:2]) - 1.) / 2.
    translation = np.zeros(3)
    translation[:2] = v

    transform = sitk.AffineTransform(3)
    transform.SetCenter(s / 2)
    transform.SetTranslation(translation)
    transform.Rotate(0, 1, -np.deg2rad(best_angle))

    # restore directional information.
    out_im = sitk.Resample(newim, transform)
    out_im.CopyInformation(image)
    return out_im, transform


def centering(inputdir ,outputdir, segdir=None, globber=None):
    """
    Center image files at their center of mass and rotate to align with coronal symmetry plane
    """
    os.makedirs(outputdir, exist_ok=True)

    segfiles = None
    if not segdir is None:
        ids = get_unique_IDs(os.listdir(segdir))
        infiles, segfiles = load_supervised_pair_by_IDs(inputdir, segdir, ids, globber=globber)
    else:
        infiles = os.listdir(inputdir)
        infiles.sort()
        if not globber is None:
            infiles = get_fnames_by_globber(infiles, globber)

    for i, f in enumerate(tqdm(infiles)):
        tqdm.write(f)
        inim_fname = inputdir + '/' + f
        inim = sitk.ReadImage(inim_fname)

        outim, transform = align_image_to_symmetry_plane(inim)
        sitk.WriteImage(sitk.Cast(outim, sitk.sitkUInt16), outputdir + '/' + f)

        if not segfiles is None:
            segim = sitk.ReadImage(segdir + '/' + segfiles[i])
            segim = transform_label(segim, transform)
            sitk.WriteImage(sitk.Cast(segim, sitk.sitkUInt8), outputdir + '/' + segfiles[i])


    pass

def normalization(inputdir, outputdir, segdir=None, globber=None, nyul_profile=None):
    """
    Head and neck normalization before benign malignant diagnosis.
    """
    abs_inputdir = os.path.abspath(inputdir)
    target_files = os.listdir(abs_inputdir)
    target_files = get_fnames_by_globber(target_files, globber)
    abs_target_files = [os.path.join(abs_inputdir, b) for b in target_files]

    # make dir to hold temp input
    os.makedirs('./.temp/nyul', exist_ok=True)
    os.makedirs('./.temp/centering', exist_ok=True)

    # step 1
    nyul(abs_target_files, './.temp/nyul', transform_file=nyul_profile)

    # step 2 & 3
    centering('./.temp/nyul', './.temp/centering', globber=globber)

    # step 4
    crop_by_directory('./temp/centering/', outputdir)

    # remove temp dir
    rmtree('./.temp')