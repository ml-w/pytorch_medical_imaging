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

    def setUp(self):
        self.subject = tio.datasets.FPG()
        self.img_dir = Path('./sample_data/img')
        self.seg_dir = Path('./sample_data/seg')
        self.temp_out_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # self.temp_out_dir.cleanup()
        MNTSLogger.cleanup()

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


class Test_visualization_rAIdiologist(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_visualization_rAIdiologist, self).__init__(*args, **kwargs)

    def setUp(self):
        self.img_dir = Path('./sample_data/img')
        self.seg_dir = Path('./sample_data/seg')
        self.json = json.load(Path('./sample_data/sample_rAIdiologist_scores.json').open('r'))
        self.image = tio.ScalarImage(str(self.img_dir.joinpath('MRI_01.nii.gz')))[tio.DATA].squeeze()
        self.prediction = np.asarray(self.json['MRI_01'])[..., 0].ravel()
        self.indices = np.asarray(self.json['MRI_01'])[..., -1].ravel()
        self.temp_out_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # self.temp_out_dir.cleanup()
        MNTSLogger.cleanup()

    def test_mark_slice(self):
        x1 = make_marked_slice(self.image[..., 15], self.prediction, self.indices)
        x2 = make_marked_slice(self.image[..., 15], self.prediction, self.indices, vert_line=15)
        x3 = make_marked_slice(self.image[..., 15], self.prediction, self.indices, imshow_kwargs={'cmap':'jet'})

        self.assertTrue(x1.shape == x2.shape == x3.shape)

    def test_mark_stack(self):
        mark_image_stacks(self.image,
                          self.prediction,
                          self.indices,
                          trim_repeats=False)

        mark_image_stacks(self.image,
                          self.prediction,
                          self.indices,
                          trim_repeats=True)

    def test_label_images_in_dir(self):
        temp_dir = tempfile.TemporaryDirectory()
        label_images_in_dir(self.img_dir, self.json, temp_dir.name, idGlobber="MRI_[0-9]+")
        print(list(Path(temp_dir.name).iterdir()))
        temp_dir.cleanup()