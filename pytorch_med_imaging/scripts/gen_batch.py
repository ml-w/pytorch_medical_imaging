from ..Algorithms.batchgenerator import *
import argparse

def console_entry(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, action='store', dest='input',
                        help='Input directory that contains nifti files or a txt file with all IDs separated by comma.')
    parser.add_argument('-o', '--output', type=str, action='store', dest='output', default= None,
                        help='Output directory to folds')
    parser.add_argument('-g', '--idglobber', action='store', default=None, dest='idglobber',
                        help='If a directory holding nii files is provided ')
    parser.add_argument('-n', '--num-workers', action='store', type=int, default=None,
                        help="Specify number of workers. If not specified, use all CPU cores.")
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
        dicom2nii(a, logger)

        GenerateTestBatch(table.index,
                          a.num_of_folds,
                          out_file_dir.__str__(),
                          stratification_class=table['Tstage'],
                          validation=len(table) // 10,
                          prefix='B'
                          )
