import pprint
from pathlib import Path
from typing import Union

import SimpleITK as sitk
import numpy as np
from mnts.mnts_logger import MNTSLogger

from pytorch_med_imaging.Algorithms.post_proc_segment import edge_smoothing, keep_n_largest_connected_body, \
    remove_small_island_2d

__all__ = ['seg_post_main', 'grow_segmentation', 'np_specific_postproc']

def seg_post_main(in_dir: Path,
                  out_dir: Path) -> None:
    r"""Post processing segmentation"""
    with MNTSLogger('pipeline.log', 'seg_post_main') as logger:
        logger.info("{:-^80}".format(" Post processing segmentation "))
        in_dir = Path(in_dir)
        out_dir = Path(out_dir)
        source = list(Path(in_dir).glob("*.nii.gz")) + list(Path(in_dir).glob("*.nii"))

        logger.debug(f"source file list: \n{pprint.pformat([str(x) for x in source])}")

        for s in source:
            logger.info(f"processing: {str(s)}")
            in_im = sitk.Cast(sitk.ReadImage(str(s)), sitk.sitkUInt8)
            out_im = edge_smoothing(in_im, 1)
            out_im = keep_n_largest_connected_body(out_im, 1)
            out_im = remove_small_island_2d(out_im, 15) # the vol_thres won't count thickness
            out_im = np_specific_postproc(out_im)
            out_fname = out_dir.joinpath(s.name)
            logger.info(f"writing to: {str(out_fname)}")
            sitk.WriteImage(out_im, str(out_fname))


def grow_segmentation(input_segment: Union[Path, str]) -> None:
    r"""Grow the segmentation using `sitk.BinaryDilate` using a kernel of [5, 5, 2]"""
    with MNTSLogger('pipeline.log', 'get_t2w_series_files') as logger:
        input_seg_dir = Path(input_segment)
        if input_seg_dir.is_file():
            input_seg_dir = [str(input_seg_dir)]
        elif input_seg_dir.is_dir():
            input_seg_dir = list(input_seg_dir.iterdir())

        for f in input_seg_dir:
            # Process only nii files
            if f.suffix.find('nii') < 0 and f.suffix.find('gz') < 0:
                continue
            logger.info(f"Growing segmentation: {str(f)}")
            seg = sitk.Cast(sitk.ReadImage(str(f)), sitk.sitkUInt8)
            seg_out = sitk.BinaryDilate(seg, [5, 5, 2])
            sitk.WriteImage(seg_out, str(f))


def np_specific_postproc(in_im: sitk.Image) -> sitk.Image:
    r"""This post-processing protocol was designed to compensate the over sensitiveness of the CNN, mainly the focus
    was given to the top two and bottom tow slices. Criteria used to remove the noise segmented by the CNN.


    Args:
        in_im (sitk.Image or str):
            Input image

    Returns:
        sitk.Image
    """
    thickness_thres = 2 # mm
    # From bottom up, opening followed by size threshold until something was left
    shape = in_im.GetSize()
    spacing = in_im.GetSpacing()
    vxel_vol = np.cumprod(spacing)[-1]

    kernel_size = (np.ones(shape=2) * thickness_thres) / np.asarray(spacing)[:2]
    kernel_size = np.ceil(kernel_size).astype('int')

    # create out image
    out_im = sitk.Cast(in_im, sitk.sitkUInt8)
    for i in range(shape[-1]):
        slice_im = out_im[:,:,i]

        # skip if sum is 0
        if np.isclose(sitk.GetArrayFromImage(slice_im).sum(), 0):
            continue

        # suppose there will only be one connected component
        filter = sitk.ConnectedComponentImageFilter()
        conn_im = filter.Execute(slice_im)
        n_objs = filter.GetObjectCount() - 1
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(conn_im)
        sizes = np.asarray([shape_stats.GetPhysicalSize(i) for i in range(1, filter.GetObjectCount() + 1)])
        keep_labels = np.argwhere(sizes >= 25) + 1 # keep only islands with area > 20mm^2

        out_slice = sitk.Image(slice_im)
        for j in range(n_objs + 1): # objects label value starts from 1
            if (j + 1) in keep_labels:
                continue
            else:
                # remove from original input if label is not kept.
                out_slice = out_slice - sitk.Mask(slice_im, conn_im == (j + 1))

        # Remove very thin segments
        out_slice = sitk.BinaryOpeningByReconstruction(out_slice, kernel_size.tolist())
        out_slice = sitk.JoinSeries(out_slice)
        out_im = sitk.Paste(out_im, out_slice, out_slice.GetSize(), destinationIndex=[0, 0, i])

        # if after processing, the slice is empty continue to work on the next slice
        if np.isclose(sitk.GetArrayFromImage(out_slice).sum(), 0):
            continue
        else:
            break

    # From top down
    for i in list(range(shape[-1]))[::-1]:
        slice_im = out_im[:,:,i]

        # skip if sum is 0
        if np.isclose(sitk.GetArrayFromImage(slice_im).sum(), 0):
            continue

        # suppose there will only be one connected component
        filter = sitk.ConnectedComponentImageFilter()
        conn_im = filter.Execute(slice_im)
        n_objs = filter.GetObjectCount() - 1
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(conn_im)
        sizes = np.asarray([shape_stats.GetPhysicalSize(i) for i in range(1, filter.GetObjectCount() + 1)])
        keep_labels = np.argwhere(sizes >= 100) + 1 # keep only when area > 100mm^2, note that no slice thickness here

        out_slice = sitk.Image(slice_im)
        for j in range(n_objs + 1): # objects label value starts from 1
            if (j + 1) in keep_labels:
                continue
            else:
                # remove from original input if label is not kept.
                out_slice = out_slice - sitk.Mask(slice_im, conn_im == (j + 1))

        out_im = sitk.Paste(out_im, out_slice, out_slice.GetSize(), destinationIndex=[0, 0, i])
        # if after processing, the slice is empty continue to work on the next slice
        if np.isclose(sitk.GetArrayFromImage(out_slice).sum(), 0):
            continue
        else:
            break

    out_im.CopyInformation(in_im)
    return out_im