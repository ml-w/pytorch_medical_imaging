from ..utils.preprocessing import make_mask_from_dir
import argparse

def console_entry_point():
    parser = argparse.ArgumentParser(
        description="Create mask using binary threshold filter for all images in a directory.")
    parser.add_argument('-i', '--input', dest='input', action='store',
                        help="Input directory, should contain .nii.gz files.")
    parser.add_argument('-o', '--output', dest='output', action='store',
                        help="Output directory where all the mask will be generated to.")
    parser.add_argument('-lt', '--lower-threshold', dest='threshold_lower', type=float, action='store',
                        help="Lower threshold.")
    parser.add_argument('-ut', '--upper-threshold', dest='threshold_upper', type=float, action='store',
                        help="Upper threshold.")
    parser.add_argument('--mask-inside', dest='mask_inside', action='store_false',
                        help="If used this flag, pixels with value inside the bound will be 0 in the output.")
    parser.add_argument('--fill-holes', dest='fill_holes', action='store_true',
                        help="If used this flag, masks are filled slice-by-slice. (Implementing)")
    a = parser.parse_args()

    make_mask_from_dir(a.input,
                       a.output,
                       a.threshold_lower,
                       a.threshold_upper,
                       a.mask_inside)