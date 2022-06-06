import unittest
import torchio as tio
from pytorch_med_imaging.utils.visualization import *

class Test_visualization(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_visualization, self).__init__(*args, **kwargs)

    def setUp(self):
        self.subject = tio.datasets.FPG()

    def test_draw_grid(self):
        img = self.subject['t1'][tio.DATA].squeeze().permute(2, 0, 1).unsqueeze(1)
        seg = self.subject['seg'][tio.DATA].squeeze().permute(2, 0, 1).unsqueeze(1)
        draw_grid(img.float(),
                  seg.int(),
                  ground_truth=None)


