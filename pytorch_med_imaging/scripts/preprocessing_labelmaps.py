from ..utils.preprocessing_labelmaps import remap_label as rl
from ..med_img_dataset import ImageDataSet
from .console_entry import pmi_console_entry
import os
import ast

__all__ = ['remap_label']

def remap_label():
    r"""Remap labels"""
    parser = pmi_console_entry()
    parser.make_console_entry_io()
    parser.add_argument('-m', '--remap', type=str, action='store',
                        help="Remap dictionary.")
    parser.add_argument('-n', '--num-worker', dest='numworker', type=int, default=8, action='store',
                        help="Number of workers.")
    args = parser.parse_args()
    assert os.path.isdir(args.output), f"Cannot open output directory: {args.output}"

    remap_dict = ast.literal_eval(args.remap)

    dataset = ImageDataSet(args.input, verbose=args.verbose, dtype='uint8', filtermode='idlist', idlist=args.idlist)
    rl(remap_dict, dataset, args.output, args.numworker)


