import os, re
import pprint
import shutil
import time

from typing import Union, Sequence
from pytorch_med_imaging.main import console_entry as pmi_main
from pytorch_med_imaging.Algorithms.post_proc_segment import main as seg_post_main
from mnts.scripts.normalization import run_graph_inference
# from pytorch_med_imaging.scripts.dicom2nii import console_entry as dicom2nii
from mnts.mnts_logger import MNTSLogger
from mnts.scripts.dicom2nii import console_entry as dicom2nii

from pathlib import Path
import argparse
import tempfile

from .report_gen import *
import SimpleITK as sitk
import pandas as pd

setting = {
    'asset_dir': Path('../asset').resolve(),
    'temp_dir': Path('./temp_dir').resolve()
}


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', action='store', type=str,
                        help="If directory is supplied, treat target as DICOM. If more than one series is found, this "
                             "program will try to identify the T2w-fs sequence, but its not always correct. If a"
                             "NIFTI file is given, the dicom2nii step is skipped")
    parser.add_argument('-o', '--output', action='store', type=str,
                        help="Where the outputs are generated. If the directory does not exist, it will be created.")
    parser.add_argument('-n', '--num-worker', action='store', type=int, default=-1,
                        help="Number of workers, if 0 or 1, some steps are ran in the main thread. If negative, all"
                             "CPUs will be utilized.")
    parser.add_argument('--verbose', action='store_true',
                        help="Verbosity.")
    parser.add_argument('--keep-log', action='store_true',
                        help='If specified the log file will be kept as pipeline.log')
    parser.add_argument('--keep-data', action='store_true',
                        help="If specified the segmentations and other images generated will be kept")
    parser.add_argument('--debug', action='store_true',
                        help="For debug.")
    a = parser.parse_args(raw_args)

    # set up logger
    with MNTSLogger('./pipline.log', 'main_report', keep_file=a.keep_log, verbose=a.verbose, log_level="debug") as logger:
        # Prepare directories
        temp_dir = tempfile.TemporaryDirectory()
        temp_dirname = Path(temp_dir.name)
        output_dir = Path(a.output)
        if not output_dir.is_dir():
            output_dir.mkdir(exist_ok=True)

        # TODO:
        # * put radiomics models into trained_states
        # * write an algorithm to make decisions based on the segmentation

        input = Path(a.input)
        if input.is_dir():
            # Get dicom files and copy to a temp directory
            _dicom_files = get_t2w_series_files(a.input)
            _temp_dicom_dir = tempfile.TemporaryDirectory()
            [shutil.copy2(d, _temp_dicom_dir.name) for d in _dicom_files]
            dicom2nii(f"-i {_temp_dicom_dir.name} -o {str(temp_dirname)} -n {a.num_worker} -g '.*' "
                      f"--dump-dicom-tags --use-patient-id".split())

        elif input.is_file() and input.suffix in ('.nii', '.gz'):
            # copy that to the temp dir
            shutil.copy(str(input), str(temp_dirname))
        else:
            raise IOError(f"Input specified is incorrect, expect a directory or an nii file, got '{a.input}' instead.")

        # Normalize target
        nii_files = list(temp_dirname.iterdir())
        normalized_dir = temp_dirname.joinpath('normalized_image_raw')
        normalized_dir.mkdir()
        if len(nii_files) == 0:
            raise FileNotFoundError(f"Nothing is found in the temporary directory.")
        run_graph_inference(f"-i {str(temp_dirname)} -o {str(normalized_dir)} "
                            f"-f ./asset/t2w_normalization.yaml --state-dir ./asset/trained_states".split())

        #!! Debug
        # normalized_dir.mkdir(exist_ok=True)
        # normalized_dir.joinpath('NyulNormalizer').mkdir(exist_ok=True)
        # normalized_dir.joinpath('HuangThresholding').mkdir(exist_ok=True)
        # shutil.copy2(str(input), str(normalized_dir.joinpath('NyulNormalizer')))
        # shutil.copy2(str(input.with_name('1183.nii.gz')), str(normalized_dir.joinpath('HuangThresholding')))

        #=================
        # Run segmentation
        #=================
        segment_output = temp_dirname.joinpath('segment_output')
        override_tags = {
            '(Data,input_dir)': str(normalized_dir.joinpath('NyulNormalizer')),
            '(Data,prob_map_dir)': str(normalized_dir.joinpath('HuangThresholding')),
            '(Data,output_dir)': str(segment_output)
        }
        override_string = ';'.join(['='.join([k, v]) for k, v in override_tags.items()])
        command = f"--config=./asset/pmi_config/NPC_seg.ini " \
                  f"--override={override_string} --inference --verbose".split()
        logger.debug(f"{command}")
        pmi_main(command)

        # Slightly grow the segmented region such that the patch sampling is done better
        grow_segmentation(str(segment_output))

        # Re-run segmentation using the previous segmentation as sampling reference.
        override_tags = {
            '(Data,input_dir)': str(normalized_dir.joinpath('NyulNormalizer')),
            '(Data,prob_map_dir)': str(segment_output),
            '(Data,output_dir)': str(segment_output)
        }
        override_string = ';'.join(['='.join([k, v]) for k, v in override_tags.items()])
        command = f"--config=./asset/pmi_config/NPC_seg.ini " \
                  f"--override={override_string} --inference --verbose".split()
        logger.debug(f"{command}")
        pmi_main(command)

        # Copy the image to temp folder
        normalized_image_dir = temp_dirname.joinpath('normalized_image')
        if normalized_image_dir.is_dir():
            shutil.rmtree(str(normalized_image_dir))
        shutil.copytree(str(normalized_dir.joinpath('NyulNormalizer')), str(normalized_image_dir))
        [shutil.copy2(str(t), str(normalized_image_dir)) for t in temp_dirname.glob('*.json')]

        # Segmentation post-processing
        command = f"-i {str(segment_output)} -o {str(segment_output)} -v".split()
        seg_post_main(command)
        if a.keep_data:
            logger.info(f"Writing data to: {str(output_dir.joinpath(segment_output.name))}")
            try:
                # Copy result to output dir
                shutil.copytree(str(segment_output), str(output_dir.joinpath(segment_output.name)))
            except Exception as e:
                logger.warning("Failed to save output!")
                logger.exception(e)


        #==============
        # Run radiomics
        #==============

        #============================
        # Run deep learning diagnosis
        #============================
        dl_output_dir = temp_dirname.joinpath('dl_diag')
        override_tags = {
            '(Data,input_dir)': str(normalized_dir.joinpath('NyulNormalizer')),
            '(Data,mask_dir)': str(segment_output),
            '(Data,output_dir)': str(dl_output_dir)
        }
        override_string = ';'.join(['='.join([k, v]) for k, v in override_tags.items()])
        command = f"--config=./asset/pmi_config/BM_nyul_v2.ini " \
                  f"--override={override_string} --inference".split()
        logger.info(f"Command for deep learning analysis: {command}")
        pmi_main(command)
        # shutil.copytree(str(dl_output_dir), str(output_dir.joinpath(dl_output_dir.name)))

        # Convert DL outputs to report gen data and run gen report
        report_dir = output_dir.joinpath('report')
        report_dir.mkdir(exist_ok=True)
        generate_report(temp_dirname, report_dir)

        temp_dir.cleanup()

def get_t2w_series_files(in_dir):
    r"""Check and see if there are T2w-fs files, if there are more than one, prompt users
    to select from a list which to use"""
    with MNTSLogger('pipeline.log', 'get_t2w_series_files') as logger:
        in_dir = Path(in_dir)
        # Check all subdirs
        _tag_list = []
        _file_list = []
        for r, d, f in os.walk(str(in_dir)):
            _curdir = Path(r)
            if not len(f) == 0:
                series_ids = sitk.ImageSeriesReader_GetGDCMSeriesIDs(str(_curdir))
                for sid in series_ids:
                    dicom_files = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(
                            str(_curdir),
                            sid
                        )
                    dicom_files = [str(Path(difil).absolute()) for difil in dicom_files]
                    logger.debug(f"Found files: {dicom_files}")

                    header_reader = sitk.ImageFileReader()
                    header_reader.SetFileName(dicom_files[0])
                    header_reader.LoadPrivateTagsOn()
                    header_reader.ReadImageInformation()
                    all_tags = {k: header_reader.GetMetaData(k) for k in header_reader.GetMetaDataKeys()}

                    tags = {
                        '0008|103e': None,  # Image description, usually they put protocol name here
                        '0018|1030': None,  # Acquisition protocol name
                        '2001|100b': None,  # Scan Plane
                    }
                    for dctag in tags:
                        try:
                            tags[dctag] = header_reader.GetMetaData(dctag).rstrip().rstrip(' ')
                        except:
                            logger.debug(f"Tag [{dctag}] missing for image {f}")
                            tags[dctag] = 'Missing'
                    logger.debug(f"Tags: {pprint.pformat(tags)}")

                    if (re.match("((?i)(?=).*T2.*)((?i)(?=).*(fs).*)|((?i)(?=).*stir.*)", tags['0018|1030']) is None) or \
                            (re.match(f"(?i)(?=).*TRA.*", tags['2001|100b']) is None):
                        logger.debug(f"Series {sid}({tags['0018|1030']}) is not T2w-fs")
                        continue

                    # If pass these two tests, return the file list
                    logger.debug(f"Find target DICOM files '{tags['0018|1030']}'")
                    _file_list.append(tuple([tags, dicom_files]))

        if len(_file_list) == 0:
            raise ArithmeticError(f"Cannot find a T2w-fs image in {in_dir}")
        elif len(_file_list) > 1:
            # force verbose
            _v = logger._verbose
            logger._verbose = True
            logger.warning(f"More than one target sequence found, please choose from below the desired sequence...")
            logger.info('\n' + '\n\n'.join([f"[{i}]: \n {pprint.pformat(_f[0], indent=4)}" for i, _f in enumerate(_file_list)]))
            _choice = input(f"Choose sequence [0 to {len(_file_list) - 1}]: ")
            logger.debug(f"Chosen sequence: {_choice}")
            file_list = _file_list[int(_choice)][1]
            # resume
            logger._verbose = _v
        else:
            file_list = _file_list[0][1]
    return file_list

def grow_segmentation(input_segment: Union[Path, str]) -> None:
    r"""Grow the segmentation using `sitk.BinaryDilate` using a kernel of [5, 5, 2]"""
    with MNTSLogger('pipeline.log', 'get_t2w_series_files') as logger:
        input_seg_dir = Path(input_segment)
        if input_seg_dir.is_file():
            input_seg_dir = [str(input_seg_dir)]
        elif input_seg_dir.is_dir():
            input_seg_dir = list(input_seg_dir.iterdir())

        for f in input_seg_dir:
            # Process only nii files
            if f.suffix.find('nii') < 0 and f.suffix.find('gz') < 0:
                continue
            logger.info(f"Growing segmentation: {str(f)}")
            seg = sitk.Cast(sitk.ReadImage(str(f)), sitk.sitkUInt8)
            seg_out = sitk.BinaryDilate(seg, [5, 5, 2])
            sitk.WriteImage(seg_out, str(f))


def np_specific_postproc(in_im: Union[sitk.Image, str]):
    r"""This post-processing protocol was designed to compensate the over sensitiveness of the CNN, mainly the focus
    was given to the top two and bottom tow slices. Criteria used to remove the noise segmented by the CNN.


    Args:
        in_im (sitk.Image or str):
            Input image

    Returns:
        sitk.Image
    """




def generate_report(root_dir: Union[Path, str], out_dir: Union[Path, str]) -> None:
    with MNTSLogger('pipeline.log', 'generate_report') as logger:
        root_dir = Path(root_dir)
        out_dir = Path(out_dir)

        # Create folders to hold the itmes
        report_dir = out_dir.joinpath('report_dir')
        report_dir.mkdir(exist_ok=True)

        res_csv = pd.read_csv(list(root_dir.joinpath('dl_diag').glob('*.csv'))[0], index_col=0)
        for i, (f_im, f_seg, f_tag) in enumerate(zip(root_dir.joinpath('normalized_image').glob("*nii*"),
                                           root_dir.joinpath('segment_output').glob("*nii*"),
                                           root_dir.joinpath('normalized_image').glob('*.json'),
                                           )):
            # Default TODO: Move this to report_gen.py
            write_out_data = {
                'ref_radiomics': '',
                'ref_dl': 0.5
            }

            # Analyse the segmentation and the output
            seg = sitk.ReadImage(str(f_seg))
            stat_fil = sitk.LabelShapeStatisticsImageFilter()
            stat_fil.Execute(seg)
            volume = stat_fil.GetPhysicalSize(1)
            dl_res = res_csv.iloc[i]['Prob_Class_0']
            id = res_csv.iloc[i]['IDs']
            if np.isreal(id):
                id = str(int(id))

            report_path = report_dir.joinpath(f'npc_report_{id}.pdf')
            c = ReportGen_NPC_Screening(str(report_path))

            write_out_data['dicom_tags'] = str(f_tag)
            write_out_data['lesion_vol'] = f'{volume / 1000.:.02f}' # convert from mm3 to cm3
            write_out_data['diagnosis_dl'] = f"{dl_res:.03f}"
            write_out_data['image_nii'] = str(f_im)
            write_out_data['segment_nii'] = str(f_seg)

            c.read_data_display(write_out_data)
            c.draw()


