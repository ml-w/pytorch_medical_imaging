import argparse
import shutil
import time
import gc

# from pytorch_med_imaging.scripts.dicom2nii import console_entry as dicom2nii
from mnts.scripts.normalization import run_graph_inference

from pytorch_med_imaging.main import console_entry as pmi_main
from .img_proc import grow_segmentation, seg_post_main, np_specific_postproc
from .report_gen import *
from .rgio import generate_id_path_map, process_input

__all__ = []

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
    parser.add_argument('-f', '--dump-diagnosis', action='store', type=str, default=None,
                        help="If specified, the resultant meta-data will be dumped to a text file. If the file exist, "
                             "it will be appended to the bottom of the file.")
    parser.add_argument('--skip-norm', action='store_true',
                        help="If specified, the program will skip the normalization step.")
    parser.add_argument('--idGlobber', action='store', type=str, default="[\w\d]+",
                        help="ID globber.")
    parser.add_argument('--skip-exist', action='store_true',
                        help="If the output report exist skip processing.")
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
    with MNTSLogger('./pipline.log', 'main_report', keep_file=a.keep_log, verbose=a.verbose, log_level="debug") as logger, \
            tempfile.TemporaryDirectory() as temp_dir:
        # Prepare directories
        temp_dirpath = Path(temp_dir)
        output_dir = Path(a.output)
        if not output_dir.is_dir():
            output_dir.mkdir(exist_ok=True)

        #╔══════╗
        #║▐ I/O ║
        #╚══════╝
        input = Path(a.input)
        process_input(a.input, temp_dir, idGlobber=a.idGlobber, num_worker=a.num_worker)

        # Check if skipping
        if a.skip_exist:
            p = temp_dirpath.glob("*nii.gz")
            mo = re.search(a.idGlobber, str(next(p).name))
            if mo is None:
                logger.warning(f"Cannot glob ID from files: {[str(pp) for pp in list(p)]}")
            else:
                id = mo.group()
                report_path = output_dir.joinpath('report/report_dir').joinpath(f'npc_report_{id}.pdf')
                if report_path.is_file():
                    logger.info(f"Skip_exist specified and target {str(report_path)} exist. Doing nothing.")
                    return

        #╔═════════════════════╗
        #║▐ Normalization      ║
        #╚═════════════════════╝
        t_0 = time.time()
        logger.info("{:-^80}".format(" Normalization "))
        nii_files = list(temp_dirpath.iterdir())
        normalized_dir = temp_dirpath.joinpath('normalized_image_raw')
        normalized_dir.mkdir()
        if len(nii_files) == 0:
            raise FileNotFoundError(f"Nothing is found in the temporary directory.")
        if not a.skip_norm:
            run_graph_inference(f"-i {str(temp_dirpath)} -o {str(normalized_dir)} "
                                f"-f ./asset/t2w_normalization.yaml --state-dir ./asset/trained_states".split())
        else:
            raise NotImplementedError("--skip-norm is not yet implemented.")
            logger.info("Skipping normalization because --skip-norm is specified.")
            logger.info("Copying source files to temp directory.")
            for i in nii_files:
                logger.info(f".. processing {str(i)}.")
                shutil.copy2(str(i), str(normalized_dir))
        logger.info("{:-^80}".format(f" Normalization Done (Total time: {time.time() - t_0:.01f}s) "))

        #!! Debug
        # normalized_dir.mkdir(exist_ok=True)
        # normalized_dir.joinpath('NyulNormalizer').mkdir(exist_ok=True)
        # normalized_dir.joinpath('HuangThresholding').mkdir(exist_ok=True)
        # shutil.copy2(str(input), str(normalized_dir.joinpath('NyulNormalizer')))
        # shutil.copy2(str(input.with_name('1183.nii.gz')), str(normalized_dir.joinpath('HuangThresholding')))

        #╔══════════════════════╗
        #║▐ Segment Primary NPC ║
        #╚══════════════════════╝
        segment_output = temp_dirpath.joinpath('segment_output')
        run_segmentation(normalized_dir, temp_dirpath, segment_output, a.idGlobber, logger)

        # Copy the image to temp folder
        normalized_image_dir = temp_dirpath.joinpath('normalized_image')
        if normalized_image_dir.is_dir():
            shutil.rmtree(str(normalized_image_dir))
        shutil.copytree(str(normalized_dir.joinpath('NyulNormalizer')), str(normalized_image_dir))
        [shutil.copy2(str(t), str(normalized_image_dir)) for t in temp_dirpath.glob('*.json')]

        # Segmentation post-processing
        # command = f"-i {str(segment_output)} -o {str(segment_output)} -v".split()
        if a.keep_data:
            # create a folder based on time at outputdir
            data_out_dir = Path(output_dir.joinpath("Saved_output"))
            shutil.rmtree(str(data_out_dir), True)
            data_out_dir.mkdir()
            logger.info(f"Writing data to: {str(data_out_dir)}")
            try:
                # Copy result to output dir
                shutil.copytree(str(segment_output), str(data_out_dir.joinpath(segment_output.name)))
                shutil.copytree(normalized_image_dir, str(data_out_dir.joinpath(normalized_image_dir.name)))
            except Exception as e:
                logger.warning("Failed to save output!")
                logger.exception(e)
        seg_post_main(segment_output, segment_output)

        #╔═════════════════════╗
        #║▐ DL Analysis        ║
        #╚═════════════════════╝
        dl_output_dir = temp_dirpath.joinpath('dl_diag')
        run_rAIdiologist(normalized_dir, segment_output, dl_output_dir, a.idGlobber, logger)
        # shutil.copytree(str(dl_output_dir), str(output_dir.joinpath(dl_output_dir.name)))

        # if the outcome is NOT cancer, but there is segmentation, run rAIdiologist again with the slices that has
        # segmentation.
        # dl_output_csv = dl_output_dir.glob('*csv')
        # im_dir = normalized_dir.joinpath('NyulNormalizer').glob('*nii.gz')
        # seg_dir = segment_output.glob("*nii.gz")
        # for i, (dl_res, _im_dir, _seg_dir) in enumerate(zip(dl_output_csv, im_dir, seg_dir)):
        #     dl_res = pd.read_csv(dl_res)
        #     # im = sitk.ReadImage(str(im))
        #     seg = sitk.ReadImage(str(_seg_dir))
        #     dl_res = dl_res.iloc[i]['Prob_Class_0']
        #     stat_fil = sitk.LabelShapeStatisticsImageFilter()
        #     stat_fil.Execute(seg)
        #     try:
        #         seg_vol = stat_fil.GetPhysicalSize(1)
        #     except RuntimeError:
        #         seg_vol = 0
        #
        #     # convert from mm3 to cm3, threshold 0.5 is same as in :func:`get_overall_diagnosis`
        #     if seg_vol / 1000 >= 0.5 and dl_res < 0.5:
        #         # Create another path, with same directory structure as the original output path
        #         out_dir     = temp_dirpath.joinpath("safety_net")
        #         dl_out_dir  = out_dir.joinpath("dl_diag")
        #         out_im_dir  = out_dir.joinpath("normalized_image/NyulNormalizer")
        #         out_seg_dir = out_dir.joinpath("segment_output")
        #         out_im_dir.mkdir(parents=True, exist_ok=True)
        #         out_seg_dir.mkdir(parents=True, exist_ok=True)
        #
        #         # [xstart, ystart, zstart, xsize, ysize, zsize]
        #         bbox   = stat_fil.GetBoundingBox(1)
        #         zstart = bbox[2]
        #         zend   = zstart + bbox[5]
        #
        #         # Give it some extra slices
        #         zstart = max(0                , zstart - 1)
        #         zend   = min(seg.GetSize()[-1], zend + 2)
        #
        #         im = sitk.ReadImage(str(_im_dir))
        #         sitk.WriteImage(im[... , zstart: zend], str(out_im_dir.joinpath(_im_dir.name)))
        #         sitk.WriteImage(seg[..., zstart: zend], str(out_seg_dir.joinpath(_seg_dir.name)))
        #
        #         run_rAIdiologist(out_dir.joinpath('normalized_image'), out_seg_dir, dl_out_dir, a.idGlobber, logger)

        run_safety_net(dl_output_dir, normalized_dir, segment_output, temp_dirpath, a.idGlobber, logger)

        #╔════════════╗
        #║ Report Gen ║
        #╚════════════╝
        # Convert DL outputs to report gen data and run gen report
        report_dir = output_dir.joinpath('report')
        report_dir.mkdir(exist_ok=True)
        generate_report(temp_dirpath, report_dir, dump_diagnosis=a.dump_diagnosis)

        logger.info("{:=^80}".format(f" Report Gen Done "))


def run_safety_net(dl_output_dir : Path,
                   normalized_dir: Path,
                   segment_output: Path,
                   temp_dirpath  : Path,
                   idGlobber     : str,
                   logger        : MNTSLogger) -> None:
    r"""Run for a second time all benign cases and record the analysis.

    This function creates the following file structure:
    .
    └── temp_dir/
        └── saftety_net/
            ├── normalized_image/NyulNormalizer/
            │   ├── cropped_slice_01.nii.gz
            │   ├── cropped_slice_02.nii.gz
            │   └── ...
            ├── segment_output/
            │   ├── cropped_segment_01.nii.gz
            │   └── cropped_segment_02.nii.gz
            └── dl_diag/
                ├── class_inf.csv
                └── class_inf.json (the risk curve)
      """
    # Read results from rAIdiologist
    dl_output_csv = pd.read_csv(str(dl_output_dir.joinpath('class_inf.csv')), index_col=0)
    dl_output_csv = dl_output_csv[dl_output_csv['Prob_Class_0'] < 0.5]
    normed_imgs = list(normalized_dir.joinpath('NyulNormalizer').glob("*nii.gz"))
    normed_id_map = {re.search(idGlobber, str(s.name)).group(): s for s in normed_imgs}
    seg_paths = list(segment_output.glob("*nii.gz"))
    seg_id_map = {re.search(idGlobber, str(s.name)).group(): s for s in seg_paths}
    safety_net_dir = temp_dirpath.joinpath("safety_net")
    safety_dl_out_dir = safety_net_dir.joinpath("dl_diag")
    safety_out_im_dir = safety_net_dir.joinpath("normalized_image/NyulNormalizer")
    safety_out_seg_dir = safety_net_dir.joinpath("segment_output")
    [_d.mkdir(parents=True, exist_ok=True) for _d in (safety_net_dir, safety_dl_out_dir,
                                                      safety_out_im_dir, safety_out_seg_dir)]
    for i, row in dl_output_csv.iterrows():
        _dl_res = row['Prob_Class_0']
        _img_path = normed_id_map.get(row.name, None)
        _seg_path = seg_id_map.get(row.name, None)
        if _img_path is None or _seg_path is None:
            logger.info(f"Paths are missing for id {row.name}")
            continue
        seg = sitk.ReadImage(str(_seg_path))
        stat_fil = sitk.LabelShapeStatisticsImageFilter()
        stat_fil.Execute(seg)
        try:
            seg_vol = stat_fil.GetPhysicalSize(1)
        except RuntimeError:  # Skip if no segmentation
            continue

        # Generate focused images if the condition met
        if seg_vol / 1000 >= 0.5:
            # [xstart, ystart, zstart, xsize, ysize, zsize]
            bbox = stat_fil.GetBoundingBox(1)
            zstart = bbox[2]
            zend = zstart + bbox[5]
            # Give it some extra slices
            zstart = max(0, zstart - 1)
            zend = min(seg.GetSize()[-1], zend + 2)
            im = sitk.ReadImage(str(_img_path))
            sitk.WriteImage(im[..., zstart: zend], str(safety_out_im_dir.joinpath(_img_path.name)))
            sitk.WriteImage(seg[..., zstart: zend], str(safety_out_seg_dir.joinpath(_seg_path.name)))

    # Only perform next step when there are benign cases because the directory will be empty otherwise.
    if len(list(safety_dl_out_dir.iterdir())) > 0:
        run_rAIdiologist(safety_net_dir.joinpath('normalized_image'),
                         safety_out_seg_dir, safety_dl_out_dir, idGlobber, logger)


def run_segmentation(normalized_dir, temp_dirname, segment_output, idGlobber, logger) -> None:
    t_0 = time.time()

    #╔══════════════════════╗
    #║▐ Coarse Segmentation ║
    #╚══════════════════════╝
    logger.info("{:-^80}".format(" Segmentation - Coarse "))
    override_tags = {
        '(Data,input_dir)': str(normalized_dir.joinpath('NyulNormalizer')),
        '(Data,prob_map_dir)': str(normalized_dir.joinpath('HuangThresholding')),
        '(Data,output_dir)': str(segment_output),
        '(LoaderParams,idGlobber)': str(idGlobber)
    }
    override_string = ';'.join(['='.join([k, v]) for k, v in override_tags.items()])
    command = f"--config=./asset/pmi_config/NPC_seg.ini " \
              f"--override={override_string} --inference --verbose".split()
    logger.debug(f"{command}")
    pmi_main(command)
    logger.info("{:-^80}".format(f" Segmentation - Coarse Done (Total time: {time.time() - t_0:.01f}s) "))
    # Slightly grow the segmented region such that the patch sampling is done better
    grow_segmentation(str(segment_output))

    #╔═════════════════════╗
    #║▐ Fine segmentation  ║
    #╚═════════════════════╝
    # Re-run segmentation using the previous segmentation as sampling reference.
    t_0 = time.time()
    logger.info("{:-^80}".format(" Segmentation - Fine "))
    override_tags = {
        '(Data,input_dir)': str(normalized_dir.joinpath('NyulNormalizer')),
        '(Data,prob_map_dir)': str(segment_output),
        '(Data,output_dir)': str(segment_output),
        '(LoaderParams,idGlobber)': str(idGlobber)
    }
    override_string = ';'.join(['='.join([k, v]) for k, v in override_tags.items()])
    command = f"--config=./asset/pmi_config/NPC_seg.ini " \
              f"--override={override_string} --inference --verbose".split()
    logger.debug(f"{command}")
    pmi_main(command)
    logger.info("{:-^80}".format(f" Segmentation - Fine Done (Total time: {time.time() - t_0:.01f}s) "))


def run_rAIdiologist(normalized_dir, segment_output, dl_output_dir, idGlobber, logger) -> str:
    t_0 = time.time()
    logger.info("{:-^80}".format(" rAIdiologist "))
    if not dl_output_dir.is_dir():
        dl_output_dir.mkdir()
    override_tags = {
        '(Data,input_dir)': str(normalized_dir.joinpath('NyulNormalizer')),
        '(Data,mask_dir)': str(segment_output),  # This is for transformation "v1_swran_transform.yaml"
        '(Data,output_dir)': str(dl_output_dir),
        '(LoaderParams,idGlobber)': str(idGlobber)
    }
    override_string = ';'.join(['='.join([k, v]) for k, v in override_tags.items()])
    command = f"--config=./asset/pmi_config/BM_rAIdiologist_nyul_v2.ini " \
              f"--override={override_string} --inference".split()
    logger.info(f"Command for deep learning analysis: {command}")
    pmi_main(command)
    logger.info("{:-^80}".format(f" rAIdiologist Done (Total time: {time.time() - t_0:.01f}s) "))
    return dl_output_dir


def generate_report(root_dir: Union[Path, str],
                    out_dir: Union[Path, str],
                    idGlobber: Optional[str] = "^[\w\d]+",
                    dump_diagnosis: Union[Path, str] = None) -> None:
    with MNTSLogger('pipeline.log', 'generate_report') as logger:
        root_dir = Path(root_dir)
        out_dir = Path(out_dir)

        # Create folders to hold the itmes
        report_dir = out_dir.joinpath('report_dir')
        report_dir.mkdir(exist_ok=True)

        # define directories
        res_csv = pd.read_csv(str(root_dir.joinpath('dl_diag/class_inf.csv')), index_col=0)
        risk_data = root_dir.joinpath('dl_diag/class_inf.json')
        risk_data = {} if not risk_data.is_file() else json.load(risk_data.open('r'))
        path_normalized_images = root_dir.joinpath('normalized_image')
        path_segment_output = root_dir.joinpath('segment_output')
        # If safety-net directory exist
        safety_net_dir = root_dir.joinpath('safety_net')
        safety_net_FLAG = len(list(safety_net_dir.rglob("*nii.gz"))) > 0
        if safety_net_FLAG:
            res_csv_sv = pd.read_csv(list(safety_net_dir.joinpath('dl_diag').glob('class_inf.csv'))[0], index_col=0)
        else:
            res_csv_sv = None

        # Construct data array
        map_normalized_images = generate_id_path_map(path_normalized_images.glob("*.nii.gz"), idGlobber, name="norm")
        map_normalized_jsons  = generate_id_path_map(path_normalized_images.glob("*.json")  , idGlobber, name="json")
        map_segment_output    = generate_id_path_map(path_segment_output.glob("*nii.gz")   , idGlobber, name="seg")
        risk_data_series      = pd.Series(risk_data, name="risk")
        maps = [res_csv['Prob_Class_0'], map_normalized_images, map_normalized_jsons, map_segment_output, risk_data_series]
        if safety_net_FLAG:
            map_safety_net_norm = generate_id_path_map(
                safety_net_dir.joinpath("normalized_image/NyulNormalizer/").glob("*nii.gz", idGlobber), name="sf_norm")
            map_safety_net_seg = generate_id_path_map(
                safety_net_dir.joinpath("segment_output/").glob("*nii.gz", idGlobber), name="sf_seg")
            map_safety_json = generate_id_path_map(
                safety_net_dir.joinpath("normalized_image/NyulNormalizer/").glob("*json", idGlobber), name="sf_json")
            safety_net_risk_data = safety_net_dir.joinpath("diag_dl/class_inf.json")
            safety_net_risk_data = {} if not safety_net_risk_data.is_file() else json.load(safety_net_risk_data.open('r'))
            safety_net_risk_data_series = pd.Series(safety_net_risk_data, "sf_risk")
            # need new keys for safety net outputs
            safety_net_prob_class = res_csv_sv['Prob_Class_0']
            safety_net_prob_class.name = "sf_Prob_Class_0"
            maps.extend([safety_net_prob_class, map_safety_net_norm,
                         map_safety_json, map_safety_net_seg, safety_net_risk_data_series])
        mapped_table = pd.concat(maps, axis=1)

        # iterate for each set of data
        overall_diagnosis = {}
        for id, row in mapped_table.iterrows():
            # Default TODO: Move this to report_gen.py
            f_im = Path(row['norm'])
            f_seg = Path(row['seg'])
            f_tag = Path(row['json'])
            f_diag_dl = row['Prob_Class_0']
            f_risk = row['risk'] if len(row['risk']) > 0 else None

            write_out_data = {
                'ref_radiomics': '',
                'ref_dl': 0.5
            }

            # Analyse the segmentation and the output
            seg = sitk.ReadImage(str(f_seg))
            stat_fil = sitk.LabelShapeStatisticsImageFilter()
            stat_fil.Execute(seg)
            try:
                volume = stat_fil.GetPhysicalSize(1)
            except:
                volume = 0
            if np.isreal(id):
                id = str(int(id))

            report_path = report_dir.joinpath(f'npc_report_{id}.pdf')
            c = ReportGen_NPC_Screening(str(report_path))
            c.set_dump_diagnosis(dump_diagnosis)

            write_out_data['dicom_tags'] = str(f_tag)
            write_out_data['lesion_vol'] = f'{volume / 1000.:.02f}' # convert from mm3 to cm3
            write_out_data['diagnosis_dl'] = f"{f_diag_dl:.03f}"
            write_out_data['image_nii'] = str(f_im)
            write_out_data['segment_nii'] = str(f_seg)
            write_out_data['risk_data'] = f_risk
            if safety_net_FLAG:
                sf_im = Path(row['sf_norm'])
                sf_seg = Path(row['sf_seg'])
                sf_tag = Path(row['sf_json'])
                sf_diag = row['sf_Prob_Class_0']
                sf_risk = row['sf_risk'] if len(row['sf_risk']) > 0 else None
                # Try to read if there are more files
                _temp_dict = {}
                # If the files are found, put it into the key "safety_net
                if all([sf_im.is_file(), sf_seg.is_file()]):
                    _temp_dict['image_nii'] = str(sf_im)
                    _temp_dict['segment_nii'] = str(sf_seg)
                    _temp_dict['diagnosis_dl'] = f"{sf_diag:.03f}"
                    _temp_dict['risk_data'] = sf_risk
                    write_out_data['safety_net'] = _temp_dict

            c.set_data_display(write_out_data)
            c.draw()
            overall_diagnosis[id] = c.diagnosis_overall

        if dump_diagnosis:
            overall_diagnosis = pd.Series(overall_diagnosis, name='Overall Diagnosis').to_frame()
            overall_diagnosis = overall_diagnosis.join(mapped_table['Prob_Class_0'])
            if safety_net_FLAG:
                overall_diagnosis = overall_diagnosis.join(mapped_table['sf_Prob_Class_0'])
            overall_diagnosis.to_csv(str(out_dir.joinpath("diagnosis.csv")))

