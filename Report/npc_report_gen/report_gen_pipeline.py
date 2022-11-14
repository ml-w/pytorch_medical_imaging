import argparse
import shutil
import distutils
import time
import gc

from mnts.scripts.normalization import run_graph_inference
import torchio as tio

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
                        help="If specified, the program will skip the normalization step. Mask is required to skip "
                             "normalization.")
    parser.add_argument('--idGlobber', action='store', type=str, default="[^\W_]+",
                        help="ID globber.")
    parser.add_argument('--idlist', action='store', default=None,
                        help="If this is provided, only images with the given IDs are processed.")
    parser.add_argument('--skip-exist', action='store_true',
                        help="If the output report exist skip processing.")
    parser.add_argument('--verbose', action='store_true',
                        help="Verbosity.")
    parser.add_argument('--keep-log', action='store_true',
                        help='If specified the log file will be kept as pipeline.log')
    parser.add_argument('--save-segment', action='store_true',
                        help="If specified, the segmentations will be saved to `output/segment_output` directory.")
    parser.add_argument('--save-normed', action='store_true',
                        help="If specified, the segmentations will be saved to `output/normalized_images` directory.")
    parser.add_argument('--keep-data', action='store_true',
                        help="If specified the segmentations and other images generated will be kept")
    parser.add_argument('--debug', action='store_true',
                        help="For debug.")
    a = parser.parse_args(raw_args)

    # set up logger
    t_start = time.time()
    with MNTSLogger('./pipline.log', 'main_report', keep_file=a.keep_log, verbose=a.verbose, log_level="debug") as logger, \
            tempfile.TemporaryDirectory() as temp_dir:
        # Prepare directories
        temp_dirpath = Path(temp_dir)
        output_dir = Path(a.output)
        if not output_dir.is_dir():
            output_dir.mkdir(exist_ok=True)
        if a.keep_data:
            a.save_segment = True
            a.save_normed = True


        #╔══════╗
        #║▐ I/O ║
        #╚══════╝
        input = Path(a.input)
        if a.idlist is not None:
            idlist = a.idlist.split(',')
        else:
            idlist = generate_id_path_map(input.glob("*nii.gz"), idGlobber=a.idGlobber)
            if len(idlist) == 0:
                idlist = None

        # Check if skipping
        if a.skip_exist:
            if idlist is not None:
                copy_list = list(idlist)
                for _id in copy_list:
                    report_path = output_dir.joinpath('report/report_dir').joinpath(f'npc_report_{_id}.pdf')
                    if report_path.is_file():
                        logger.info(f"Skip_exist specified and target {str(report_path)} exist. Doing nothing.")
                        idlist.remove(_id)
                if len(idlist) == 0:
                    logger.info("Nothing left to process, terminating...")
                    return
        avail_idlist = process_input(a.input, temp_dir, idGlobber=a.idGlobber, idlist=idlist, num_worker=a.num_worker)

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
        #
        # #╔══════════════════════╗
        # #║▐ Segment Primary NPC ║
        # #╚══════════════════════╝
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
        seg_post_main(segment_output, segment_output)

        #╔═════════════════════╗
        #║▐ DL Analysis        ║
        #╚═════════════════════╝
        dl_output_dir = temp_dirpath.joinpath('dl_diag')
        run_rAIdiologist(normalized_dir, segment_output, dl_output_dir, a.idGlobber, logger)
        # shutil.copytree(str(dl_output_dir), str(output_dir.joinpath(dl_output_dir.name)))

        run_safety_net(dl_output_dir, normalized_dir, segment_output, temp_dirpath, a.idGlobber, logger)

        #╔════════════╗
        #║ Report Gen ║
        #╚════════════╝
        # Convert DL outputs to report gen data and run gen report
        report_dir = output_dir.joinpath('report')
        report_dir.mkdir(exist_ok=True)
        generate_report(temp_dirpath, report_dir, idGlobber=a.idGlobber, dump_diagnosis=a.dump_diagnosis)

        logger.info("{:=^80}".format(f" Report Gen Done (Total: {time.time() - t_start:.01f}s) "))

        #╔══════════════╗
        #║▐ Save output ║
        #╚══════════════╝
        if a.save_segment:
            save_dir = output_dir.joinpath("segment_output")
            if not save_dir.is_dir():
                save_dir.mkdir(parents=True)
            src_dir = segment_output
            distutils.dir_util.copy_tree(str(src_dir), str(save_dir), preserve_mode=0)
            logger.info(f"Saving segmentation files to {str(save_dir)}")

        if a.save_normed:
            save_dir = output_dir.joinpath("normalized_images")
            if not save_dir.is_dir():
                save_dir.mkdir(parents=True)
            src_dir = normalized_dir
            distutils.dir_util.copy_tree(str(src_dir), str(save_dir), preserve_mode=0)
            logger.info(f"Saving normalized image files to {str(save_dir)}")

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
    # Added range for focal evaluation
    offset = 2

    # Read results from rAIdiologist
    dl_output_csv = pd.read_csv(str(dl_output_dir.joinpath('class_inf.csv')), index_col=0)
    dl_output_csv.index = dl_output_csv.index.astype(str)
    dl_output_csv = dl_output_csv[dl_output_csv['Prob_Class_0'] < 0.5]
    normed_imgs = list(normalized_dir.joinpath('NyulNormalizer').glob("*nii.gz"))
    normed_id_map = {re.search(idGlobber, str(s.name)).group(): s for s in normed_imgs}
    seg_paths = list(segment_output.glob("*nii.gz"))
    seg_id_map = {re.search(idGlobber, str(s.name)).group(): s for s in seg_paths}
    safety_net_FLAG = False
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
            zstart = max(0, zstart - offset)
            zend = min(seg.GetSize()[-1], zend + offset + 1)
            im = sitk.ReadImage(str(_img_path))
            sitk.WriteImage(im[..., zstart: zend], str(safety_out_im_dir.joinpath(_img_path.name)))
            sitk.WriteImage(seg[..., zstart: zend], str(safety_out_seg_dir.joinpath(_seg_path.name)))
            safety_net_FLAG = True

    # Only perform next step when there are benign cases because the directory will be empty otherwise.
    if len(dl_output_csv) > 0 and safety_net_FLAG:
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

    # sometimes the coarse segmentation results in nothing when there are no thickening. Temporally disable
    # them by removing the *nii.gz
    changed = []
    for f_seg in segment_output.glob("*nii???"):
        # check if label is empty
        seg_im = tio.LabelMap(f_seg)
        if seg_im.count_nonzero() == 0:
            logger.warning(f"{f_seg} found to have no coarse segmentatino. Skipping it.")
            img_link = normalized_dir.joinpath('NyulNormalizer').joinpath(f_seg.name)
            new_name = Path(re.sub("\.nii(\.gz)?", '', str(img_link)))
            changed.append((img_link.absolute(), new_name.absolute()))
            logger.debug(f"{str(img_link.absolute())} renamed to {str(new_name.absolute())}")
            img_link.replace(new_name)

    #╔═════════════════════╗
    #║▐ Fine segmentation  ║
    #╚═════════════════════╝
    # Re-run segmentation using the previous segmentation as sampling reference.
    if len(changed) < len(list(normalized_dir.joinpath('NyulNormalizer').iterdir())):
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
    else:
        logger.info("Skipping fine segmentation because coarse segmentation resulted in nothing.")

    # change back the names:
    for ori, now in changed:
        now.replace(ori)

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
                    idGlobber: Optional[str] = "^[^\W_]+",
                    dump_diagnosis: Union[Path, str] = None) -> None:
    with MNTSLogger('pipeline.log', 'generate_report') as logger:
        root_dir = Path(root_dir)
        out_dir = Path(out_dir)

        # Create folders to hold the itmes
        report_dir = out_dir.joinpath('report_dir')
        report_dir.mkdir(exist_ok=True)

        # define directories
        res_csv = pd.read_csv(str(root_dir.joinpath('dl_diag/class_inf.csv')), index_col=0)
        res_csv.index = res_csv.index.astype(str)
        risk_data = root_dir.joinpath('dl_diag/class_inf.json')
        risk_data = {} if not risk_data.is_file() else json.load(risk_data.open('r'))
        path_normalized_images = root_dir.joinpath('normalized_image')
        path_segment_output = root_dir.joinpath('segment_output')
        # If safety-net directory exist11
        safety_net_dir = root_dir.joinpath('safety_net')
        safety_net_FLAG = len(list(safety_net_dir.rglob("*nii.gz"))) > 0
        if safety_net_FLAG:
            safety_net_csv = pd.read_csv(list(safety_net_dir.joinpath('dl_diag').glob('class_inf.csv'))[0], index_col=0)
            safety_net_csv.index = safety_net_csv.index.astype(str)
        else:
            safety_net_csv = None

        # Construct data array
        map_normalized_images = generate_id_path_map(path_normalized_images.glob("*.nii.gz"), idGlobber, name="norm")
        map_normalized_jsons  = generate_id_path_map(path_normalized_images.glob("*.json")  , idGlobber, name="json")
        map_segment_output    = generate_id_path_map(path_segment_output.glob("*nii.gz")   , idGlobber, name="seg")
        risk_data_series      = pd.Series(data=risk_data, name="risk")
        maps = [res_csv['Prob_Class_0'], map_normalized_images, map_normalized_jsons, map_segment_output, risk_data_series]
        if safety_net_FLAG:
            map_safety_net_norm = generate_id_path_map(
                safety_net_dir.joinpath("normalized_image/NyulNormalizer/").glob("*nii.gz"), idGlobber, name="sf_norm")
            map_safety_net_seg = generate_id_path_map(
                safety_net_dir.joinpath("segment_output/").glob("*nii.gz"), idGlobber, name="sf_seg")
            safety_net_risk_data = safety_net_dir.joinpath("dl_diag/class_inf.json")
            safety_net_risk_data = {} if not safety_net_risk_data.is_file() else json.load(safety_net_risk_data.open('r'))
            safety_net_risk_data_series = pd.Series(data=safety_net_risk_data, name="sf_risk")
            # need new keys for safety net outputs
            safety_net_prob_class = safety_net_csv['Prob_Class_0']
            safety_net_prob_class.name = "sf_Prob_Class_0"
            maps.extend([safety_net_prob_class, map_safety_net_norm,
                         map_safety_net_seg, safety_net_risk_data_series])
        for _df in maps:
            if any(_df.duplicated()):
                logger.warning(f"Duplicate found in {_df.to_string()}."
                               f"Trying to drop but results might be wrong.")
                _df.drop_duplicate(inplace=True)


        mapped_table = pd.concat(maps, axis=1)

        # iterate for each set of data
        overall_diagnosis = {}
        for id, row in mapped_table.iterrows():
            try:
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
                if safety_net_FLAG and not pd.isna(row['sf_norm']):
                    sf_im = Path(row['sf_norm'])
                    sf_seg = Path(row['sf_seg'])
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
            except Exception as e:
                msg = f"Error when dealing with file with id: {id}\n" \
                      f"{row}"
                logger.error(msg)
                logger.exception(e)
                continue

        if dump_diagnosis:
            overall_diagnosis = pd.Series(overall_diagnosis, name='Overall Diagnosis').to_frame()
            overall_diagnosis = overall_diagnosis.join(mapped_table['Prob_Class_0'])
            if safety_net_FLAG:
                try:
                    overall_diagnosis = overall_diagnosis.join(mapped_table['sf_Prob_Class_0'])
                except Exception as e:
                    logger.warning("Attach safety net results failed. Skipping.")
                    logger.debug(f"Original error: {e}")

            csv_path =  out_dir.joinpath("diagnosis.csv")
            overall_diagnosis.to_csv(str(csv_path), mode='a', header=not csv_path.is_file())

