import unittest
import os
import subprocess
import tempfile
import shutil
from pytorch_med_imaging.main import console_entry as pmi_main
from npc_report_gen.report_gen_pipeline import get_t2w_series_files, main
from pathlib import Path

class Test_pipeline(unittest.TestCase):
    def test_get_t2w_series(self):
        p = Path("example_data/npc_case/ALL_DICOM")
        files = get_t2w_series_files(str(p.absolute()))
        self.assertEqual(len(files) > 0, True)

    def test_main(self):
        # p = Path("example_data/npc_case/ALL_DICOM")
        p = Path("/media/storage/Source/Repos/NPC_Segmentation/NPC_Segmentation/00.RAW/HKU/0135")
        # p = Path("example_data/npc_case/1183-T2_FS_TRA+301.nii.gz")
        with tempfile.TemporaryDirectory() as temp_dir:
            args = f"-i {str(p)} -o {str(p.parent)} -n 16 --verbose"
            main(args.split())

    def test_grow_segmentation(self):
        p = Path("")

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

if __name__ == '__main__':
    diu = Test_pipeline()
    diu.test_main()