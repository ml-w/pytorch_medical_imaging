import unittest
import torchio as tio
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path
from pytorch_med_imaging.utils.visualization.segmentation_vis import *
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

    def test_draw_contour(self):
        img = self.subject['t1'][tio.DATA].squeeze().permute(2, 0, 1).float().numpy()
        seg = self.subject['seg'][tio.DATA].squeeze().permute(2, 0, 1).int().numpy()

        cont = draw_contour(img[125].T,
                            seg[125].T,
                            contour_alpha=0.2,
                            contour_thickness=1)

    def test_draw_contour_with_crop(self):
        img = self.subject['t1'][tio.DATA].squeeze().permute(2, 0, 1).float().numpy()
        seg = self.subject['seg'][tio.DATA].squeeze().permute(2, 0, 1).int().numpy()

        cont = draw_contour(img[125].T,
                            seg[125].T == 1,
                            crop = True,
                            crop_padding = 10,
                            contour_alpha=0.2,
                            contour_thickness=1)

        cont = draw_contour(img[125].T,
                            seg[125].T == 1,
                            seg[125].T == 4,
                            crop = True,
                            crop_padding = 10,
                            contour_alpha=0.2,
                            contour_thickness=1)
        plt.imshow(cont)
        plt.show()

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

