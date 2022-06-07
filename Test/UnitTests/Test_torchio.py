import unittest
import tempfile
import yaml
import torchio as tio
from pytorch_med_imaging.PMI_data_loader.augmenter_factory import create_transform_compose


class TestTorchIO(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTorchIO, self).__init__(*args, **kwargs)

    def setUp(self) -> None:
        # try loading transforms
        self.transform_1 = \
            """
            ToCanonical:
                {}
            RandomAffine:
                scales: [0.9, 1.1]
                degrees: [0, 0, 10]
            RandomFlip:
                - 'lr'
            RandomNoise: 
                std: [0, 8]
            CropOrPad:
                - [350, 350, 40]
                - padding_mode: 0
                - crop_pad_sides: cct    
            """
        with tempfile.NamedTemporaryFile('w+', suffix='yaml') as f:
            f.write(self.transform_1)
            f.flush()
            self.augment = create_transform_compose(f.name)

        self.sample = tio.Subject(image=tio.ScalarImage('./sample_data/img/sample_1.nii.gz'))
        pass

    def test_transform(self):
        transformed = self.augment.apply_transform(self.sample)
        print(transformed[0])

