from eros import *
from tqdm import *
import SimpleITK as sitk
import numpy as np
import re
import sys, os
from utils import *

def transform_label(label_im, transform):
    assert isinstance(label_im, sitk.Image)
    assert isinstance(transform, sitk.AffineTransform)

    tempim = sitk.GetImageFromArray(sitk.GetArrayFromImage(label_im))

    out_im = sitk.Resample(tempim, transform)
    out_im.CopyInformation(label_im)
    return out_im


def compute_com(im):
    """compute_com -> (int, int, int)
    Compute the center of mass of the input image.
    """

    filter = sitk.LabelShapeStatisticsImageFilter()
    filter.Execute(im)

    try:
        com = filter.GetCentroid(filter.GetLabels()[0])
        com = im.TransformPhysicalPointToIndex(com)
    except IndexError as e:
        print("Enconter error: ", e)
        raise IndexError(e)
    # if isinstance(im, sitk.Image):
    #     im = sitk.GetArrayFromImage(im)
    #
    # im_shape = im.shape
    # im_grid = np.meshgrid(*[np.arange(i) for i in im_shape], indexing='ij')
    #
    # z, x, y= [im * im_grid[i] for i in range(3)]
    # z, x, y= [X.sum() for X in [x, y, z]]
    # z, x, y= [X / float(np.sum(im)) for X in [x, y, z]]
    # z, x, y= np.round([x, y, z]).astype('int')
    # print(z, x, y)
    return com

def align_image_to_symmetry_plane(image, center = None):
    assert isinstance(image, sitk.Image)

    ssfactor    = 4
    eros_res    = eros.eros(sitk.GetArrayFromImage(image)[:,::ssfactor,::ssfactor], 2, angle_range=[-10, 10])
    best_angle  = eros_res.get_mean_angle()
    com         = eros_res.get_mean_com() * ssfactor if center is None else center

    # strip directional information
    npim = sitk.GetArrayFromImage(image)
    newim = sitk.GetImageFromArray(npim)
    s = np.array(newim.GetSize())

    # perform shift
    v =  np.array(com) - (np.array(image.GetSize()[:2]) - 1.) / 2.
    translation = np.zeros(3)
    translation[:2] = v

    transform = sitk.AffineTransform(3)
    transform.SetCenter(s / 2)
    transform.SetTranslation(translation)
    transform.Rotate(0, 1, -np.deg2rad(best_angle))

    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(npim.min())
    resampler.SetTransform(transform)
    resampler.SetSize(image.GetSize())
    out_im = resampler.Execute(newim)
    # out_im = sitk.Resample(newim, transform, defaultPixelValue=npim.min())
    out_im.CopyInformation(image)
    return out_im, transform

def crop_image(im, center, size):
    in_imsize = im.GetSize()
    lower_bound = [int(c - s // 2) for c, s in zip(center, size)]
    # upper_bound = [int(ori_s - c - np.ceil(s / 2.)) for c, s, ori_s in zip(center, size, in_imsize)]

    # Check bounds
    max_x, max_y, max_z = (np.array(in_imsize) - np.array(size))
    lower_bound[0] = np.clip(lower_bound[0], 0, max_x)
    lower_bound[1] = np.clip(lower_bound[1], 0, max_y)
    lower_bound[2] = np.clip(lower_bound[2], 0, max_z)
    lower_bound = [int(l) for l in lower_bound]
    upper_bound = [ori_s - lb - s for ori_s, lb, s in zip(in_imsize, lower_bound, size)]
    cropper = sitk.CropImageFilter()
    cropper.SetLowerBoundaryCropSize(lower_bound)
    cropper.SetUpperBoundaryCropSize(upper_bound)

    outim = cropper.Execute(im)
    return outim

def crop_by_directory(src_dir, out_dir, crop_size = [444,444,20], idlist = None, center=None):
    assert src_dir != out_dir, "Please select a different output directory, it cannot be the same as " \
                               "the input directory."
    os.makedirs(out_dir, exist_ok=True)

    # Read images from source dir
    files = os.listdir(src_dir)
    if not idlist is None:
        files = get_fnames_by_IDs(files, idlist)

    # Set bound to cropping regions
    min_x, min_y = np.array(crop_size[:2]) / 2

    for f in tqdm(files):
        im = sitk.ReadImage(os.path.join(src_dir, f))
        im_shape = list(im.GetSize())[:2]
        max_x, max_y = im_shape[0] - min_x, im_shape[1] - min_y

        # obtain com
        x, y, z = compute_com(im) if center is None else center

        center = [np.clip(x, min_x, max_x),
                  np.clip(y, min_y, max_y), z]

        # crop centered at COM
        cropped = crop_image(im, center, crop_size)

        # save image as the same name as the input in the output dir.
        sitk.WriteImage(cropped, os.path.join(out_dir, f))

def main(inputdir ,outputdir, idlist, segdir, globber=None):
    os.makedirs(outputdir, exist_ok=True)
    os.makedirs(outputdir+"_cropped", exist_ok=True)

    crop_size = [444, 444, 20]


    # load segmentations for computing image crop center
    ids = open(idlist, 'r').readline().rstrip().split(',')
    infiles, segfiles = load_supervised_pair_by_IDs(inputdir, segdir, ids)

    # load images to rotate and center
    coms = []
    for seg in tqdm(segfiles, desc="Compute COM"):
        try:
            coms.append(compute_com(sitk.ReadImage(os.path.join(segdir, seg))))
        except IndexError as e:
            print("Error in computing com for {}".format(seg))
            print(e)
            coms.append(None)


    for i, f in enumerate(tqdm(infiles, desc="Aligning")):
        tqdm.write(f)
        inim_fname = inputdir + '/' + f
        inim = sitk.ReadImage(inim_fname)

        # Z-score normalization
        inim = sitk.Normalize(sitk.Cast(inim, sitk.sitkFloat64))
        inim = sitk.ShiftScale(inim, scale=256)
        center = coms[i]
        if center is None:
            continue
        # outim, transform = align_image_to_symmetry_plane(inim, center = center[:2])
        outim, transform = align_image_to_symmetry_plane(inim, center = None)
        sitk.WriteImage(sitk.Cast(outim, sitk.sitkFloat32), outputdir + '/' + f)

        im_size = outim.GetSize()
        crop_center = [im_size[0] / 2, im_size[1] / 2, center[2]]
        cropped = crop_image(outim, crop_center, crop_size)
        sitk.WriteImage(sitk.Cast(cropped, sitk.sitkFloat32), outputdir + '_cropped/' + f)


    pass

if __name__ == '__main__':
    # main('../NPC_Segmentation/41.Benign/T2WFS/',
    #      '../NPC_Segmentation/42.Benign_upright/T2WFS')
    main('../NPC_Segmentation/0A.NIFTI_ALL/Malignant/CE-T1W_TRA/',
         '../NPC_Segmentation/50.NPC_SurvivalAnalysis/CE-T1W_TRA/Images_Upright',
         '../NPC_Segmentation/99.Testing/Survival_analysis/all_case.txt',
         '../NPC_Segmentation/98.Output/Survival_analysis_seg')
    # '../NPC_Segmentation/21.NPC_Perfect_SegT2/00.First',
    # globber="(?=.*T2.*)(?=.*FS.*)(?!.*[cC].*)")




