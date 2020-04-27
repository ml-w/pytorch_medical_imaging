import SimpleITK as sitk
import os
import numpy as np
import pandas as pd
from tqdm import *

from utils import get_fnames_by_IDs


def crop_image(im, center, size):
    in_imsize = im.GetSize()
    lower_bound = [int(c - s // 2) for c, s in zip(center, size)]
    upper_bound = [int(ori_s - c - np.ceil(s / 2.)) for c, s, ori_s in zip(center, size, in_imsize)]

    print(lower_bound, upper_bound, center)
    cropper = sitk.CropImageFilter()
    cropper.SetLowerBoundaryCropSize(lower_bound)
    cropper.SetUpperBoundaryCropSize(upper_bound)

    outim = cropper.Execute(im)
    return outim


def compute_com(im):
    """compute_com -> (int, int, int)
    Compute the center of mass of the input image.
    """

    if isinstance(im, sitk.Image):
        im = sitk.GetArrayFromImage(im)

    im_shape = im.shape
    im_grid = np.meshgrid(*[np.arange(i) for i in im_shape], indexing='ij')

    z, x, y= [im * im_grid[i] for i in range(3)]
    z, x, y= [X.sum() for X in [x, y, z]]
    z, x, y= [X / float(np.sum(im)) for X in [x, y, z]]
    z, x, y= np.round([x, y, z]).astype('int')
    return z, x, y


def crop_by_directory(src_dir, out_dir, crop_size = [444,444,20], idlist = None):
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
        z, x, y = compute_com(im)

        center = [np.clip(x, min_x, max_x),
                  np.clip(y, min_y, max_y), z]

        # crop centered at COM
        cropped = crop_image(im, center, crop_size)

        # save image as the same name as the input in the output dir.
        sitk.WriteImage(cropped, os.path.join(out_dir, f))
