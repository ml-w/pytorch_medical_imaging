import os
import shutil
import tempfile
import unittest
import pandas as pd
from pathlib import Path

from mnts.mnts_logger import MNTSLogger

from npc_report_gen.img_proc import seg_post_main
from npc_report_gen.report_gen_pipeline import generate_report, main
from npc_report_gen.rgio import get_t2w_series_files, process_input, generate_id_path_map
# from pytorch_med_imaging.Algorithms.post_proc_segment import main as seg_post_main
from pytorch_med_imaging.main import console_entry as pmi_main


class Test_pipeline(unittest.TestCase):
    def setUp(self):
        self._logger = MNTSLogger('.', 'test_report', verbose=True, keep_file=False, log_level='debug')
        self.temp_output_path = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        # self._logger.cleanup()
        self.temp_output_path.cleanup()

    def tearDownClass() -> None:
        # make sure the log file is cleaned properly
        MNTSLogger.cleanup()

    def test_get_t2w_series(self):
        p = Path("test_data/npc_case/ALL_DICOM")
        files = get_t2w_series_files(str(p.absolute()))
        self.assertEqual(len(files) > 0, True)

    def test_process_input(self):
        # input is a dir
        temp_input = tempfile.TemporaryDirectory()
        temp_out_path = Path(self.temp_output_path.name)
        tfs = [tempfile.NamedTemporaryFile(dir=temp_input.name, suffix='.nii.gz') for i in range(5)]
        process_input(temp_input.name, temp_out_path)
        self.assertTrue(len(list(temp_out_path.glob("*nii.gz"))) > 0,
                        f"Nothing was created: {list(temp_out_path.iterdir())}")
        self.assertTrue(all([_tf.is_symlink() for _tf in temp_out_path.glob("*.nii.gz")]),
                        f"Some of the file are not symlinks: {list(temp_out_path.iterdir())}")

        # input is a single nifty file
        named_file = list(Path(temp_input.name).glob("*.nii.gz"))[0]
        temp_out_dir2 = tempfile.TemporaryDirectory()
        temp_out_path2 = Path(temp_out_dir2.name)
        process_input(named_file, temp_out_path2)
        self.assertTrue(temp_out_path2.joinpath(named_file.name).is_symlink())

        # input is a DICOM dir is tested elsewhere

        # cleanup
        [t.close() for t in tfs]
        temp_out_dir2.cleanup()

    @unittest.SkipTest
    def test_main(self):
        p = Path("./test_data/npc_case/NIFTI/img")
        po = Path(self.temp_output_path.name)
        pof = po.joinpath('diag.csv')
        # for pp in p.glob("*nii.gz"):
        #     try:
        main(['-i',
              str(p),
              '-o',
              str(po),
              '-n',
              '16',
              '-f',
              str(pof),
              '--verbose',
              '--keep-log',
              ])
        # except Exception as e:
        #     self._logger.exception(e)
        #     self.fail(f"Something went wrong for {str(pp)}")

    def test_post_proc_main(self):
        p = Path("test_data/npc_case/NIFTI/seg/eg01.nii.gz")

        with tempfile.TemporaryDirectory() as temp_dir:
            _in = Path(temp_dir).joinpath('input')
            _out = Path(temp_dir).joinpath('output')
            _in.mkdir()
            _out.mkdir()
            shutil.copy2(str(p), str(_in))

            seg_post_main(_in, _out)

            self.assertTrue(len(os.listdir(str(_out))) > 0)

    def test_segmentation(self):
        p = Path('test_data/npc_case/')
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_dir = Path(temp_dir).joinpath('mask')
            img_dir = Path(temp_dir).joinpath('img')
            mask_dir.mkdir(exist_ok=True)
            img_dir.mkdir(exist_ok=True)
            shutil.copy2(str(p.joinpath('NIFTI/seg/eg01.nii.gz')), str(mask_dir))
            shutil.copy2(str(p.joinpath('NIFTI/seg/eg02.nii.gz')), str(mask_dir))
            shutil.copy2(str(p.joinpath('NIFTI/img/eg01.nii.gz')), str(img_dir))
            shutil.copy2(str(p.joinpath('NIFTI/img/eg02.nii.gz')), str(img_dir))

            override_tags = {
                '(Data,input_dir)': str(img_dir),
                '(Data,prob_map_dir)': str(mask_dir),
                '(Data,output_dir)': str(Path(temp_dir).joinpath('output'))
            }
            override_string = ';'.join(['='.join([k, v]) for k, v in override_tags.items()])
            command = f"--config=./asset/pmi_config/NPC_seg.ini " \
                      f"--override={override_string} --inference --verbose".split()
            pmi_main(command)

            self.assertGreater(len(list(Path(temp_dir).joinpath('output').iterdir())), 0)

    def test_dl_diag(self):
        p = Path('test_data/npc_case/')
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_dir = Path(temp_dir).joinpath('mask')
            img_dir = Path(temp_dir).joinpath('img')
            mask_dir.mkdir(exist_ok=True)
            img_dir.mkdir(exist_ok=True)
            shutil.copy2(str(p.joinpath('NIFTI/seg/eg01.nii.gz')), str(mask_dir))
            shutil.copy2(str(p.joinpath('NIFTI/img/eg01.nii.gz')), str(img_dir))

            override_tags = {
                '(Data,input_dir)': str(img_dir),
                '(Data,mask_dir)': str(mask_dir),
                '(Data,output_dir)': str(Path(temp_dir).joinpath('output'))
            }
            override_string = ';'.join(['='.join([k, v]) for k, v in override_tags.items()])
            command = f"--config=./asset/pmi_config/BM_rAIdiologist_nyul_v2.ini " \
                      f"--override={override_string} --inference --verbose".split()
            pmi_main(command)
            print(list(Path(temp_dir).joinpath("output").joinpath('class_inf.csv').open('r').readlines()))

    def test_process_output(self):
        p = Path('./test_data/report_gen')
        generate_report(p, self.temp_output_path.name)

    def test_generate_id_path_map(self):
        p = Path("./test_data/npc_case/NIFTI/img/")
        idGlobber = "^[\w\d]+"
        id_map = generate_id_path_map(p.glob("*nii.gz"), idGlobber)
        expected = pd.Series(data = [str(s) for s in Path("./test_data/npc_case/NIFTI/img/").glob("*nii.gz")],
                             index = [f"eg{i+1:02d}" for i in range(len(list(p.glob("*nii.gz"))))])
        self.assertTrue(expected.equals(id_map),
                        f"Expected: \n{expected}\nGot: \n{id_map}")


    def test_skip_if_exist(self):
        pass

    def test_skip_normalization(self):
        pass


if __name__ == '__main__':
    diu = Test_pipeline()
    diu.test_main()



