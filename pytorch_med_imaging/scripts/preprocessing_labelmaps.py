from ..utils.preprocessing_labelmaps import remap_label as rl
from ..med_img_dataset import ImageDataSet
import argparse
import os
import ast

__all__ = ['remap_label']

def remap_label():
    r"""Remap labels"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', action='store',
                        help="Input directory.")
    parser.add_argument('-o', '--output', action='store',
                        help="Output directory where all the generated results are stored.")
    parser.add_argument('-m', '--remap', type=str, action='store',
                        help="Remap dictionary.")
    parser.add_argument('-n', '--num-worker', dest='numworker', type=int, default=8, action='store',
                        help="Number of workers.")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Verbosity.")
    args = parser.parse_args()

    # check if output directory exist, creat if not
    if not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)
        assert os.path.isdir(args.output), f"Cannot open output directory: {args.output}"

    remap_dict = ast.literal_eval(args.remap)

    dataset = ImageDataSet(args.input, verbose=args.verbose, dtype='uint8')

    rl(remap_dict, dataset, args.output, args.numworker)


