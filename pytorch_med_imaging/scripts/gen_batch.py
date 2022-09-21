from ..Algorithms.batchgenerator import *
import argparse
import pandas as pd
from mnts.mnts_logger import MNTSLogger

def console_entry(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, action='store', dest='input',
                        help='Input directory that contains nifti files or a txt file with all IDs separated by comma.')
    parser.add_argument('-o', '--output', type=str, action='store', dest='output', default= None,
                        help='Output directory to folds')
    parser.add_argument('-g', '--idglobber', action='store', default=None, dest='idglobber',
                        help='If a directory holding nii files is provided ')
    parser.add_argument('-t', '--validation', action='store', default=None, type=float,
                        help="Set validation percentage.")
    parser.add_argument('-m', '--num-of-folds', action='store', type=int, required=True,
                        help="Number of folds to create. If 1, its a train-test split.")
    parser.add_argument('--stratification', action='store', type=str, default=None)
    parser.add_argument('--debug', action='store_true',
                        help="Debug mode.")
    parser.add_argument('--prefix', default="", type=str,
                        help="Add a preffix to the fold ini/txt files.")
    parser.add_argument('--log', action='store_true', dest='log',
                        help='Keep log file')
    a = parser.parse_args(raw_args)

    with MNTSLogger('./default.log', logger_name='dicom2nii', verbose=True, keep_file=a.log) as logger:
        logger.info("Recieve argumetns: {}".format(a))
        if a.validation is not None:
            assert 0 < a.validation < 1, "Option --validaiton must be between 0 and 1."

        # if input is excel or csv
        input_dir = Path(a.input)
        output_dir = Path(a.output)
        if input_dir.suffix in ('.csv', '.xlsx'):
            # take first column as ids
            if input_dir.suffix == 'csv':
                table = pd.read_csv(str(input_dir.resolve()), index_col=0)
            else:
                table = pd.read_excel(str(input_dir.resolve()), index_col=0)
            logger.info(f"Read: \n {table.to_string()}")
        else:
            raise AttributeError(f"Only able to process csv or excel files currently, got: {input_dir.suffix}.")

        GenerateTestBatch(table.index,
                          a.num_of_folds,
                          output_dir.__str__(),
                          stratification_class=table[a.stratification],
                          validation=int(len(table) * a.validation) if not a.validation is None else None,
                          prefix=a.prefix
                          )
