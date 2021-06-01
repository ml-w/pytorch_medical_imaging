from ..utils.preprocessing_labelmaps import remap_label as rl, label_statistics
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



def pmi_label_statistics():
    parser = pmi_console_entry('iOgLnv')
    parser.add_argument('--normalize', action='store_true', help="Normalize the pixel count.")
    args = parser.parse_args()

    df = label_statistics(args.input, args.idglobber, args.numworker, verbose=args.verbose, normalized=args.normalize)
    if args.outfile.endswith('.csv'):
        df.to_csv(args.outfile)
    else:
        df.to_excel(args.outfile)