from mnts.mnts_logger import MNTSLogger
from pytorch_med_imaging.utils import preprocessing, data_formatting
import argparse
import os

__all__ = ['dicom2nii']

def dicom2nii(a, logger):
    if not os.path.isdir(a.root_dir):
        raise FileNotFoundError(f"Cannot open {a.root_dir}.")
    if not os.path.isdir(a.output):
        logger.warning(f"Target directory does not exist, trying to create: {a.output}")
        os.makedirs(a.output, exist_ok=True)
        if not os.path.isdir(a.output):
            logger.error("Error making output directory.")
            return

    logger.info(f"Specified ID globber: {a.idglobber}")
    logger.info(f"Use patient ID: {a.usepid}")
    ids = a.idlist.split(',') if not a.idlist is None else None
    dicom_dirs = preprocessing.recursive_list_dir(a.depth, a.root_dir)
    logger.info(f"Dirs:\n{dicom_dirs}")

    data_formatting.batch_dicom2nii(dicom_dirs,
                                    out_dir = a.output,
                                    workers = a.num_workers,
                                    seq_filters = None,
                                    idglobber = a.idglobber,
                                    use_patient_id = a.usepid,
                                    use_top_level_fname = a.usefname,
                                    input = a.root_dir,
                                    idlist = ids,
                                    prefix = a.prefix,
                                    debug = a.debug_mode,
                                    dump_meta_data = a.dump_dicom_tags)

def console_entry(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, action='store', dest='input',
                        help='Input directory that contains the DICOMs.')
    parser.add_argument('-o', '--output', type=str, action='store', dest='output', default= None,
                        help='Output directory to hold the nii files.')
    parser.add_argument('-d', '--depth', type=int, action='store', default=3, dest='depth',
                        help='Depth of DICOM file search.')
    parser.add_argument('-g', '--idglobber', action='store', default=None, dest='idglobber',
                        help='Specify the globber to glob the ID from the DICOM paths.')
    parser.add_argument('-n', '--num-workers', type=int, default=None,
                        help="Specify number of workers. If not specified, use all CPU cores.")
    parser.add_argument('--dump-dicom-tags', action='store_true',
                        help="If this option is specified, the dicom tags will be generated to a json text file.")
    parser.add_argument('--debug', action='store_true',
                        help="Debug mode.")
    parser.add_argument('--prefix', default="", type=str,
                        help="Add a preffix to the patient's ID.")
    parser.add_argument('--idlist', action='store', dest='idlist', default=None, type=str,
                        help='Only do conversion if the globbed ID is in the list. e.g. ["ID1", "ID2", ...]')
    parser.add_argument('--use-top-level-fname', action='store_true', dest='usefname',
                        help='Use top level file name immediately after the input directory as ID.')
    parser.add_argument('--use-patient-id', action='store_true', dest='usepid',
                        help='Use patient id as file id.')
    parser.add_argument('--log', action='store_true', dest='log',
                        help='Keep log file under ./dicom2nii.log')
    a = parser.parse_args(raw_args)

    with MNTSLogger('./dicom2nii.log', logger_name='dicom2nii', verbose=True, keep_file=a.log) as logger:
        logger.info("Recieve argumetns: {}".format(a))
        dicom2nii(a, logger)

if __name__ == '__main__':
    console_entry()