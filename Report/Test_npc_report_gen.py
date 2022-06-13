import unittest
import os
import subprocess
import tempfile
import shutil
from pytorch_med_imaging.Algorithms.post_proc_segment import main as seg_post_main
from pytorch_med_imaging.main import console_entry as pmi_main
from npc_report_gen.report_gen_pipeline import get_t2w_series_files, main, generate_report, seg_post_main
from pathlib import Path

class Test_pipeline(unittest.TestCase):
    def test_get_t2w_series(self):
        p = Path("example_data/npc_case/ALL_DICOM")
        files = get_t2w_series_files(str(p.absolute()))
        self.assertEqual(len(files) > 0, True)

    def test_main(self):
        # p = Path("example_data/npc_case/ALL_DICOM")
        # p = Path("/media/storage/Source/Repos/NPC_Segmentation/NPC_Segmentation/00.RAW/extra_20210426/T1rhoNPC020")
        # p = Path("/mnt/ftp_shared/NPC_New_case/2005-2021_not into T1rho studies/1719")
        # pp = Path("/mnt/ftp_shared/NPC_Screening_3_plain/Images/P385")
        # p = Path("/home/lwong/FTP/2.Projects/8.NPC_Segmentation/00.RAW/NPC_new_dx_cases/1249/")
        # p = Path("example_data/npc_case/1183-T2_FS_TRA+301.nii.gz")
        # p = Path("/media/storage/Source/Repos/NPC_Segmentation/NPC_Segmentation/00.RAW/HKU/0140")
        # p = Path('/media/storage/Data/NPC_Segmentation/70.Screening_report/TestInput/')
        p = Path('/home/lwong/Desktop/NPC_Segmentation/NPC_Segmentation/0A.NIFTI_ALL/HKU/temp/T2WFS_TRA')
        # p = Path('/home/lwong/Desktop/NPC_Segmentation/NPC_Segmentation/0A.NIFTI_ALL/All/T2WFS_TRA')
        po = Path('/media/storage/Data/NPC_Segmentation/70.Screening_report/TestOutput/')
        pof = Path('/media/storage/Data/NPC_Segmentation/70.Screening_report/TestOutput/diag.csv')
        for pp in p.glob("*177*nii.gz"):
            try:
                main(['-i',
                      str(pp),
                      '-o',
                      str(po),
                      '-n',
                      '16',
                      '-f',
                      str(pof),
                      '--verbose',
                      '--keep-data',
                      '--keep-log',
                      ])
            except:
                print(f"Something went wrong for {str(pp)}")

    def test_post_proc_main(self):
        import time
        p = Path("example_data/npc_case/1183.nii.gz")

        with tempfile.TemporaryDirectory() as temp_dir:
            _in = Path(temp_dir).joinpath('input')
            _out = Path(temp_dir).joinpath('output')
            _in.mkdir()
            _out.mkdir()
            shutil.copy2(str(p), str(_in))

            # command = f"-i {str(_in)} -o {str(_out)} -v".split()
            seg_post_main(_in, _out)

            self.assertTrue(len(os.listdir(str(_out))) > 0)

    def test_segmentation(self):
        p = Path('example_data/npc_case/')
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_dir = Path(temp_dir).joinpath('mask')
            img_dir = Path(temp_dir).joinpath('img')
            mask_dir.mkdir(exist_ok=True)
            img_dir.mkdir(exist_ok=True)
            shutil.copy2(str(p.joinpath('1183.nii.gz')), str(mask_dir))
            shutil.copy2(str(p.joinpath('1183-T2_FS_TRA+301.nii.gz')), str(img_dir))

            override_tags = {
                '(Data,input_dir)': str(img_dir),
                '(Data,prob_map_dir)': str(mask_dir),
                '(Data,output_dir)': str(Path(temp_dir).joinpath('output'))
            }
            override_string = ';'.join(['='.join([k, v]) for k, v in override_tags.items()])
            command = f"--config=./asset/pmi_config/NPC_seg.ini " \
              f"--override={override_string} --inference --verbose".split()
            pmi_main(command)

    def test_dl_diag(self):
        p = Path('example_data/npc_case/')
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_dir = Path(temp_dir).joinpath('mask')
            img_dir = Path(temp_dir).joinpath('img')
            mask_dir.mkdir(exist_ok=True)
            img_dir.mkdir(exist_ok=True)
            shutil.copy2(str(p.joinpath('1183.nii.gz')), str(mask_dir))
            shutil.copy2(str(p.joinpath('1183-T2_FS_TRA+301.nii.gz')), str(img_dir))

            override_tags = {
                '(Data,input_dir)': str(img_dir),
                '(Data,mask_dir)': str(mask_dir),
                '(Data,output_dir)': str(Path(temp_dir).joinpath('output'))
            }
            override_string = ';'.join(['='.join([k, v]) for k, v in override_tags.items()])
            command = f"--config=./asset/pmi_config/BM_nyul_v2.ini " \
              f"--override={override_string} --inference --verbose".split()
            pmi_main(command)
            print(list(Path(temp_dir).joinpath("output").joinpath('class_inf.csv').open('r').readlines()))

    def test_process_output(self):
        p = Path('/media/storage/Source/Repos/NPC_Segmentation/NPC_Segmentation/00.RAW/HKU/report/data_dir')
        generate_report(p, p)


if __name__ == '__main__':
    diu = Test_pipeline()
    diu.test_main()