from pytorch_med_imaging.logger import Logger
from pytorch_med_imaging.utils import preprocessing, data_formatting
import argparse
import os

def main(a, logger):
    if not os.path.isdir(a.input):
        raise FileNotFoundError(f"Cannot open {a.input}.")
    if not os.path.isdir(a.output):
        logger.warning(f"Target directory does not exist, trying to create: {a.output}")
        os.makedirs(a.output, exist_ok=True)
        if not os.path.isdir(a.output):
            logger.error("Error making output directory.")
            return

    logger.info(f"idglobber: {a.idglobber}")
    dicom_dirs = preprocessing.recursive_list_dir(a.depth, a.input)
    data_formatting.batch_dicom2nii(dicom_dirs,
                                    a.output,
                                    None,
                                    a.idglobber,
                                    a.usepid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, action='store', dest='input',
                        help='Input directory that contains the DICOMs.')
    parser.add_argument('-o', '--output', type=str, action='store', dest='output',
                        help='Output directory to hold the nii files.')
    parser.add_argument('-d', '--depth', type=int, action='store', default=3, dest='depth',
                        help='Depth of DICOM file search.')
    parser.add_argument('-g', '--idglobber', action='store', default=None, dest='idglobber',
                        help='Specify the globber to glob the ID from the DICOM paths.')
    parser.add_argument('--use-patient-id', action='store_true', dest='usepid',
                        help='Use patient id as file id.')
    parser.add_argument('--log')
    a = parser.parse_args()

    logger = Logger('./dicom2nii.log', logger_name='dicom2nii', verbose=True)
    logger.info("Recieve argumetns: {}".format(a))

    main(a, logger)

