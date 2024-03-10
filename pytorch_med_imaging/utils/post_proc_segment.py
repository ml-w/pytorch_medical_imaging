import SimpleITK as sitk
import numpy as np
import os
from .uid_ops import get_fnames_by_IDs
from mnts.mnts_logger import MNTSLogger
from typing import Union, Optional
import argparse
import sys
import fnmatch

__all__ = ['keep_n_largest_connected_body', 'edge_smoothing', 'main', 'remove_small_island_2d']

def keep_n_largest_connected_body(in_im: Union[sitk.Image,str],
                                  n: Optional[int] = 1):
    r"""
    This function cast the input into UInt8 label, extract the largest connected component and return the
    results.

    Args:
        in_im (sitk.Image or str):
            Input image or directory to input image.

    Returns:
        sitk.Image
    """
    # check if input is str, if so, load data
    if isinstance(in_im, str):
        assert os.path.isfile(in_im), "Cannot open supplied input."
        in_im = sitk.ReadImage(in_im)

    # cast image into integer
    out_im = sitk.Cast(in_im, sitk.sitkUInt8)

    # extract largest connected body
    filter = sitk.ConnectedComponentImageFilter()
    conn_im = filter.Execute(out_im)
    n_objs = filter.GetObjectCount() - 1  # 0 also counted

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(conn_im)
    sizes = [shape_stats.GetPhysicalSize(i) for i in range(1, filter.GetObjectCount() + 1)]
    sizes_rank = np.argsort(sizes)[::-1] # descending order
    keep_labels = sizes_rank[:n] + 1

    # copy image information
    out_im = sitk.Image(in_im)
    for i in range(n_objs + 1): # objects label value starts from 1
        if (i + 1) in keep_labels:
            continue
        else:
            # remove from original input if label is not kept.
            out_im = out_im - sitk.Mask(out_im, conn_im == (i + 1))

    # Make output binary if only largest is extracted.
    if n == 1:
        out_im = out_im != 0
    return out_im

def remove_small_island_2d(in_im: Union[sitk.Image, str],
                           area_thres: float
                           ):
    r"""
    This function cast the input into UInt8 label,
    Args:
        in_im (sitk.Image):
            Input segmentation. This should be of type sitkUint8.
        area_thres (float):
            Area threshold of which island with a smaller area than this will be removed from the slice.

    Returns:

    """
    shape = in_im.GetSize()
    spacing = in_im.GetSpacing()

    # kernel_size = (np.ones(shape=2) * thickness_thres) / np.asarray(spacing)[:2]
    # kernel_size = np.ceil(kernel_size).astype('int')

    out_im = sitk.Image(*([in_im.GetSize()] + [sitk.sitkUInt8]))
    # Scan the whole volume slice by slice
    for i in range(shape[-1]):
        slice_im = in_im[:,:,i]

        # skip if sum is 0
        if np.isclose(sitk.GetArrayFromImage(slice_im).sum(), 0):
            continue

        # extract largest connected body
        filter = sitk.ConnectedComponentImageFilter()
        conn_im = filter.Execute(slice_im)
        n_objs = filter.GetObjectCount() - 1  # 0 also counted

        # copy image information
        out_slice = sitk.Image(slice_im)
        if n_objs > 0:
            shape_stats = sitk.LabelShapeStatisticsImageFilter()
            shape_stats.Execute(conn_im)
            sizes = np.asarray([shape_stats.GetPhysicalSize(i) for i in range(1, filter.GetObjectCount() + 1)])
            keep_labels = np.argwhere(sizes >= area_thres) + 1

            for j in range(n_objs + 1): # objects label value starts from 1
                if (j + 1) in keep_labels:
                    continue
                else:
                    # remove from original input if label is not kept.
                    out_slice = out_slice - sitk.Mask(slice_im, conn_im == (j + 1))

        # out_slice = sitk.BinaryMorphologicalOpening(out_slice, kernel_size.tolist())
        out_slice = sitk.JoinSeries(out_slice)
        out_im = sitk.Paste(out_im, out_slice, out_slice.GetSize(), destinationIndex=[0, 0, i])
    out_im.CopyInformation(in_im)
    return out_im


def edge_smoothing(in_im: Union[sitk.Image, str],
                   radius: float):
    r"""
    This function smooth the edge of the binary label using closing and opening operation.

    Args:
        in_im (sitk.Image or str):
            Input segmentation
        radius

    Returns:

    """
    # check if input is str, if so, load data
    if isinstance(in_im, str):
        assert os.path.isfile(in_im), "Cannot open supplied input."
        in_im = sitk.ReadImage(in_im)

    if not isinstance(radius, list) or not isinstance(radius, int):
        radius = [radius, radius, 0]

    out_im = sitk.Image(*([in_im.GetSize()] + [sitk.sitkUInt8]))
    for i in range(in_im.GetSize()[-1]):
        im_slice = in_im[:,:,i]

        # skip if sum is 0
        if np.isclose(sitk.GetArrayFromImage(im_slice).sum(), 0):
            continue

        open_filter = sitk.BinaryMorphologicalOpeningImageFilter()
        open_filter.SetKernelType(sitk.sitkBall)
        open_filter.SetKernelRadius(radius)
        im_slice = open_filter.Execute(im_slice)

        clo_filter = sitk.BinaryMorphologicalClosingImageFilter()
        clo_filter.SetKernelType(sitk.sitkBall)
        clo_filter.SetKernelRadius(radius)
        out_slice = clo_filter.Execute(im_slice)
        out_slice = sitk.JoinSeries(out_slice)
        out_im = sitk.Paste(out_im, out_slice, out_slice.GetSize(), destinationIndex=[0, 0, i])
    out_im.CopyInformation(in_im)
    return out_im



def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', action='store', dest='input',
                        help="Input directory.")
    parser.add_argument('-o', '--output', action='store', dest='output',
                        help="Output directory.")
    parser.add_argument('-r', '--recursive', action='store_true', dest='recursive',
                        help="Recursively load all .nii.gz files under the directory.")
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                        help="Print verbose progress.")
    parser.add_argument('-s', '--smoothing', action='store', default=None, dest='smooth', type=int, required=True,
                        help="Smoothing parameter.")
    parser.add_argument('-n', '--connected-components', action='store', default=None, type=int, required=True,
                        help="How many largest components to keep")
    parser.add_argument('--idlist', action='store', default=None, dest='idlist',
                        help='Read id from a txt file.')
    args = parser.parse_args(raw_args)


    if not os.path.isdir(args.output):
        print(f"Output directory doesn't exist, trying to creat: {args.output}")
        os.makedirs(args.output, exist_ok=True)

    print("Creating logger...")
    if MNTSLogger.global_logger is None:
        logger1 = MNTSLogger(logger_name='post-processing', log_dir=os.path.join(args.output, 'post-processing.log'),
                         verbose=args.verbose)
        logger = logger1
    else:
        logger = MNTSLogger['post-processing']

    # Hook execptions
    logger.info("{:=^120}".format(" Post-processing "))

    # get all .nii.gz files
    if args.recursive:
        filelist = []
        for r, d, f in os.walk(args.input):
            if not len(f) == 0:
                for ff in f:
                    if ff.find('.nii.gz') >= 0:
                        filelist.append(os.path.join(r, ff))
    else:
        filelist = os.listdir(args.input)
        filelist = fnmatch.filter(filelist, '*.nii.gz')
        filelist = [os.path.join(args.input, f) for f in filelist]

    logger.debug("Filelist found: \n{}".format('\n'.join(filelist)))

    # filter by idlist
    if not args.idlist is None:
        filelist = get_fnames_by_IDs(filelist, args.idlist)
        filelist = [filelist[f] for f in filelist] # dict to list

    for f in filelist:
        logger.info(f"Processing {f}")

        try:
            out_im = edge_smoothing(sitk.Cast(sitk.ReadImage(f), sitk.sitkUInt8), args.smooth)
            out_im = keep_n_largest_connected_body(out_im, a.connected_components)
        except Exception:
            logger.exception(f"Error occured for {f}")

        out_fname = os.path.join(args.output, os.path.basename(f))
        sitk.WriteImage(out_im, os.path.join(args.output, os.path.basename(f)))
        logger.info(f"Writing to {out_fname}")

#
# if __name__ == '__main__':
#     test_arguments = ['-i', '/home/lwong/Source/Repos/NPC_Segmentation/NPC_Segmentation/00.RAW/HKU/segment_output',
#                       '-o', '/home/lwong/Source/Repos/NPC_Segmentation/NPC_Segmentation/00.RAW/HKU/segment_output_2',
#                       '-v']
#
#     main(test_arguments)