from eros import *
from tqdm import *
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from utils import *

def transform_label(label_im, transform):
    assert isinstance(label_im, sitk.Image)
    assert isinstance(transform, sitk.AffineTransform)

    tempim = sitk.GetImageFromArray(sitk.GetArrayFromImage(label_im))

    out_im = sitk.Resample(tempim, transform)
    out_im.CopyInformation(label_im)
    return out_im


def align_image_to_symmetry_plane(image):
    assert isinstance(image, sitk.Image)

    ssfactor    = 4
    # map = sitk.MaximumProjection(image, 2)
    eros_res    = eros.eros(sitk.GetArrayFromImage(image)[:,::ssfactor,::ssfactor], 2, angle_range=[-10, 10])
    best_angle  = eros_res.get_mean_angle()
    com         = eros_res.get_mean_com() * ssfactor
    print(com)

    # best_angle, com = eros.eros(sitk.GetArrayFromImage(map)[:,::ssfactor,::ssfactor], 2, angle_range=[-10, 10])[0]
    #
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

    out_im = sitk.Resample(newim, transform)
    out_im.CopyInformation(image)
    return out_im, transform


def main(inputdir ,outputdir, segdir=None, globber=None):
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

if __name__ == '__main__':
    # main('../NPC_Segmentation/41.Benign/T2WFS/',
    #      '../NPC_Segmentation/42.Benign_upright/T2WFS')
    main('../NPC_Segmentation/41.Benign_Malignant/',
         '../NPC_Segmentation/41.Benign_Malignant_Upright')
         # '../NPC_Segmentation/21.NPC_Perfect_SegT2/00.First',
         # globber="(?=.*T2.*)(?=.*FS.*)(?!.*[cC].*)")




