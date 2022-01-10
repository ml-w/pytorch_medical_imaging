import SimpleITK as sitk
import numpy as np
import os
from pytorch_med_imaging.logger import Logger
from pytorch_med_imaging.Algorithms.utils import get_fnames_by_IDs
import argparse
import sys
import fnmatch

__all__ = ['keep_n_largest_connected_body', 'edge_smoothing', 'main']

def keep_n_largest_connected_body(in_im: sitk.Image or str, n: int = 1):
    r"""
    This function cast the input into binary label, extract the largest connected component and return the
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
    for i in range(n_objs):
        if (i + 1) in keep_labels:
            continue
        else:
            # remove from original input if label is not kept.
            out_im = out_im - sitk.Mask(out_im, conn_im == (i + 1))

    # Make output binary if only largest is extracted.
    if n == 1:
        out_im = out_im != 0
    return out_im


def edge_smoothing(in_im: sitk.Image or str, radius):
    r"""
    This function smooth the edge of the binary label using closing and opening operation.

    Args:
        in_im:

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
    parser.add_argument('-s', '--smoothing', action='store', default=1, dest='smooth', type=int,
                        help="Smoothing parameter.")
    parser.add_argument('--idlist', action='store', default=None, dest='idlist',
                        help='Read id from a txt file.')
    args = parser.parse_args(raw_args)


    if not os.path.isdir(args.output):
        print(f"Output directory doesn't exist, trying to creat: {args.output}")
        os.makedirs(args.output, exist_ok=True)

    print("Creating logger...")
    if Logger.global_logger is None:
        logger = Logger(logger_name='post-processing', log_dir=os.path.join(args.output, 'post-processing.log'),
                        verbose=args.verbose)
    else:
        logger = Logger['post-processing']

    # Hook execptions
    sys.excepthook = logger.exception_hook

    logger.info("=" * 20 + " Post-processing " + "=" * 20)

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
            out_im = keep_n_largest_connected_body(f, 1)
            out_im = edge_smoothing(out_im, args.smooth)
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