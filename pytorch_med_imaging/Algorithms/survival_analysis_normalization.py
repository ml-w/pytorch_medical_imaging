from eros import *
from tqdm import *
import SimpleITK as sitk
import numpy as np
import sys, os
import re
from utils import *
from pytorch_med_imaging.logger import Logger

global _logger


def transform_image(im: sitk.Image,
                    transform: sitk.AffineTransform):
    r"""
    Wrapped SITK function to transform the image but keep the original space definition. Use with causion because
    this breaks the physical space of the input image (equivalent to the patient moved but the FOV didn't).
    """
    tempim = sitk.GetImageFromArray(sitk.GetArrayFromImage(im))

    out_im = sitk.Resample(tempim, transform)
    out_im.CopyInformation(im)
    return out_im


def compute_com(im: sitk.Image):
    r"""
    Wrapped SITK function for extracting the COM from the label image.

    Args:
        im (sitk.Image): Label image for locating COM

    Returns:
        com (float, float, float)
    """

    filter = sitk.LabelShapeStatisticsImageFilter()
    filter.Execute(im)

    try:
        com = filter.GetCentroid(filter.GetLabels()[0])
        # com = im.TransformPhysicalPointToIndex(com)
    except IndexError as e:
        _logger.exception("Enconter error: {}".format(e))
        raise IndexError(e)
    return com

def align_image_to_symmetry_plane(image: sitk.Image,
                                  center: (int, int, int)):
    r"""
    Use the EROS method to find the symmatry center of a 3D image by average the symmetry line of each of its
    2D slices and then averaging them. Rotate about index center.
    """
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

def crop_image(im: sitk.Image,
               center: (int, int, int) or (float, float, float),
               size: (int, int, int),
               center_is_physical: bool = False):
    # if center is physical position instead of the index, trasnform it into index first
    if center_is_physical:
        # crop at center of the axial slices
        center = np.asarray(im.TransformPhysicalPointToIndex(center))
    # Crop at center for x, y, but the z center depends on the input com.
    fov_center = np.asarray(im.GetSize()) // 2
    center[0] = fov_center[0]
    center[1] = fov_center[1]
    _logger.debug(f"center: {center}, fov_center: {fov_center}")


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

def crop_image_wrt_label(im: sitk.Image,
                         center: tuple or list,
                         size: tuple or list,
                         label: sitk.Image):
    in_imsize = im.GetSize()
    lower_bound = [int(c - s // 2) for c, s in zip(center, size)]
    # upper_bound = [int(ori_s - c - np.ceil(s / 2.)) for c, s, ori_s in zip(center, size, in_imsize)]

    # find top and bottom slices that have labels
    label_np = sitk.GetArrayFromImage(label)
    label_np  = label_np.sum(axis=[1, 2])
    label_vect = np.argwhere(label_np > 0)
    top, bot = label_vect.max(), label_vect.top()

    # Check bounds
    max_x, max_y, max_z = (np.array(in_imsize) - np.array(size))
    lower_bound[0] = np.clip(lower_bound[0], 0, max_x)
    lower_bound[1] = np.clip(lower_bound[1], 0, max_y)
    lower_bound[2] = np.clip(lower_bound[2], 0, max_z)
    lower_bound = [int(l) for l in lower_bound]
    upper_bound = [ori_s - lb - s for ori_s, lb, s in zip(in_imsize, lower_bound, size)]
    lower_bound[2] = bot
    upper_bound[2] = in_imsize[2] - top
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


def main(source_output_pair: [(str, str), (str, str), ...],
         idlist: [str, str, ...] or str,
         segdir: str,
         crop_by_labels: bool = False,
         id_globber="[0-9]{3,5}"):
    for source, output in source_output_pair:
        os.makedirs(output , exist_ok=True)
        os.makedirs(output + "_cropped", exist_ok=True)
        os.makedirs(output + "_cropped_masked", exist_ok=True)

    crop_size = [444, 444, 20]


    # load segmentations for computing image crop center, eros align based on first in dict
    first_input = source_output_pair[0][0]
    first_output = source_output_pair[0][1]
    seg_out_dir = first_output + '/../../Segmentation'
    seg_out_dir_cropped = first_output + "/../../Segmentation_cropped"
    os.makedirs(seg_out_dir, exist_ok = True)
    os.makedirs(seg_out_dir_cropped, exist_ok = True)

    if isinstance(idlist, list):
        ids = [str(l) for l in idlist]
    elif os.path.isfile(idlist):
        ids = [r.rstrip() for r in open(idlist, 'r').readlines()]
    else:
        _logger.error("ID not provided.")
        return

    # ids = ids[:3]
    _logger.info(f"Recieved idlist: {ids}")
    infiles, segfiles = load_supervised_pair_by_IDs(first_input, segdir, ids)
    _logger.debug(f"infile: {len(infiles)} seg_files: {len(segfiles)}")

    # load images to rotate and center
    coms = {}
    for seg in tqdm(segfiles, desc="Compute COM"):
        # Verify the ID
        _id = re.search(id_globber, seg)
        if _id is None:
            _logger.warning(f"Cannot glob id for {seg}, skipping...")
            continue
        else:
            _id = _id.group()

        try:
            coms[_id] = compute_com(sitk.ReadImage(os.path.join(segdir, seg)))
        except IndexError as e:
            _logger.exception("Error in computing com for {}".format(seg))
            _logger.exception("Skipping...")
            coms[_id] = None


    # Normalize the first pair of supplied directories
    _logger.info("{:-^100}".format(" Step 1: Normalizing 1st pair "))
    a_transforms = {}
    segs = []
    for i, (f,s) in enumerate(tqdm(zip(infiles, segfiles), desc="Aligning", total=len(infiles))):
        _logger.info(f"Processing {f}")
        inim_fname = first_input + '/' + f
        inim = sitk.ReadImage(inim_fname)

        # because some ID on the idlist might be missing, make sure the ID is right here.
        _id = re.search(id_globber, f)
        if _id is None:
            _logger.warning(f"Cannot glob id for {f}, skipping...")
            continue
        else:
            _id = _id.group()

        # Z-score normalization
        inim = sitk.Normalize(sitk.Cast(inim, sitk.sitkFloat64))
        inim = sitk.ShiftScale(inim, scale=256)

        # Get COM
        try:
            seg_im = sitk.ReadImage(os.path.join(segdir, s))
            com = compute_com(seg_im)
            coms[_id] = com
        except IndexError as e:
            _logger.exception("Error in computing COM for {}".format(s))
            _logger.warning(f"Skipping {s} ...")
            coms[_id] = None

        center = com
        if center is None:
            a_transforms[_id] = None
            continue

        # Trasnform the image
        outim, transform = align_image_to_symmetry_plane(inim, center = None)

        # Transform label as well if its supplied
        try:
            segout = transform_image(seg_im, transform)
        except Exception as e:
            _logger.exception("Cannot transform seg image")
            _logger.warning("Skipping...")

        # Write image and label. These are the upright images.
        sitk.WriteImage(sitk.Cast(outim, sitk.sitkFloat32), first_output + '/' + f)
        sitk.WriteImage(sitk.Cast(segout, sitk.sitkUInt8), seg_out_dir + '/' + f)

        im_size = outim.GetSize()

        # Either cropped by given size or by labels
        crop_center = [im_size[0] / 2, im_size[1] / 2, center[2]]
        if not crop_by_labels:
            cropped = crop_image(outim, com, crop_size, center_is_physical=True)
            sDegcropped = crop_image(segout, com, crop_size, center_is_physical=True)
        else:
            cropped = crop_image_wrt_label(outim, crop_center, crop_size, segout)
            segcropped = crop_image_wrt_label(segout, crop_center, crop_size, segout)

        segcropped = sitk.Cast(segcropped, sitk.sitkUInt8)
        segs.append(segcropped)
        sitk.WriteImage(sitk.Cast(cropped, sitk.sitkFloat32), first_output + '_cropped/' + f)
        sitk.WriteImage(segcropped, seg_out_dir_cropped + '/' + f)

        segcropped.CopyInformation(cropped)
        masked = sitk.Mask(cropped, segcropped)
        sitk.WriteImage(sitk.Cast(masked, sitk.sitkFloat32), first_output + '_cropped_masked/' + f)

        # record transform for the rest of the images pairs.
        a_transforms[_id] = transform

    _logger.info("{:-^100}".format(" Step 2: Normalizing remaining pairs "))
    for i, (source, output) in enumerate(tqdm(source_output_pair, total=len(source_output_pair), position=0)):
        if source == first_input:
            continue

        imfiles, _ = load_supervised_pair_by_IDs(source, segdir, ids)
        for j, f in enumerate(tqdm(imfiles, desc="Aligning", position=1)):
            # verify id
            _id = re.search(id_globber, f)
            if _id is None:
                _logger.warning(f"Cannot glob id for {f}, skipping...")
                continue
            else:
                _id = _id.group()

            _logger.info(f"Processing {f}")
            inim_fname = source + '/' + f
            inim = sitk.ReadImage(inim_fname)

            center = coms.get(_id, None)

            if center is None:
                _logger.warning(f"COM of {f} was None. Skipping")
                continue

            try:
                # Z-score normalization
                inim = sitk.Normalize(sitk.Cast(inim, sitk.sitkFloat64))
                inim = sitk.ShiftScale(inim, scale=256)

                transform = a_transforms[_id]
                if transform is None:
                    continue
                outim = transform_image(inim, transform)
                sitk.WriteImage(sitk.Cast(outim, sitk.sitkFloat32), os.path.join(output, f))

                im_size = outim.GetSize()
                crop_center = [im_size[0] / 2, im_size[1] / 2, center[2]]
                cropped = crop_image(outim, coms[_id], crop_size, center_is_physical=True)
                sitk.WriteImage(sitk.Cast(cropped, sitk.sitkFloat32), output + '_cropped/' + f)

                segs[j].CopyInformation(cropped)
                masked = sitk.Mask(cropped, segs[j])
                sitk.WriteImage(sitk.Cast(masked, sitk.sitkFloat32), output + '_cropped_masked/' + f)
            except:
                _logger.exception(f"Error processing {f}. Skipping")
                continue


    pass

if __name__ == '__main__':
    so_pair = [
        ('../../NPC_Segmentation/0A.NIFTI_ALL/Malignant/CE-T1WFS_TRA/',
         '../../NPC_Segmentation/50.NPC_SurvivalAnalysis/CE-T1WFS_TRA/Images_Upright_segonly'),
        ('../../NPC_Segmentation/0A.NIFTI_ALL/Malignant/T2WFS_TRA/',
         '../../NPC_Segmentation/50.NPC_SurvivalAnalysis/T2WFS_TRA/Images_Upright_segonly'),
        ('../../NPC_Segmentation/0A.NIFTI_ALL/Malignant/CE-T1W_TRA/',
         '../../NPC_Segmentation/50.NPC_SurvivalAnalysis/CE-T1W_TRA/Images_Upright_segonly')
    ]

    _logger = Logger('./survival_analysis_normalization.log', logger_name='Survival Normalization', verbose=True)
    try:
        _logger.info("{:=^100}".format(" Survival Analysis Normalization - Initialize "))
        main(so_pair,
             '../../NPC_Segmentation/99.Testing/Survival_analysis/all_case.txt',
             # ['835','1389'],
             '../../NPC_Segmentation/0B.Segmentations/CE-T1WFS_TRA/00.First')
        _logger.info("{:=^100}".format(" Survival Analysis Normalization - Finished "))
    except Exception as e:
        _logger.exception(e)



