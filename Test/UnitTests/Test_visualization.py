import unittest
import torchio as tio
import tempfile
from pathlib import Path
from pytorch_med_imaging.utils.visualization import *
from pytorch_med_imaging.utils.visualization_rAIdiologist import *
from mnts.mnts_logger import MNTSLogger

class Test_visualization(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_visualization, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls._logger = MNTSLogger('.', logger_name=cls.__name__, verbose=True, keep_file=False, log_level='debug')

    @classmethod
    def tearDownClass(cls) -> None:
        MNTSLogger.cleanup()

    def setUp(self):
        self.subject = tio.datasets.FPG()
        self.img_dir = Path('./sample_data/img')
        self.seg_dir = Path('./sample_data/seg')
        self.temp_out_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_out_dir.cleanup()
        # MNTSLogger.cleanup()

    def test_draw_grid(self):
        img = self.subject['t1'][tio.DATA].squeeze().permute(2, 0, 1).unsqueeze(1)
        seg = self.subject['seg'][tio.DATA].squeeze().permute(2, 0, 1).unsqueeze(1)
        draw_grid(img.float(),
                  seg.int(),
                  ground_truth=None)

    def test_draw_grid_for_dir(self):
        contour_grid_by_dir(str(self.img_dir),
                            str(self.seg_dir),
                            self.temp_out_dir.name)

    def test_draw_grid_with_crop(self):
        pass

    def test_draw_grid_with_shift(self):
        pass

