import unittest
import tempfile
import yaml
import torchio as tio
from pytorch_med_imaging.pmi_data_loader.augmenter_factory import create_transform_compose
from pytorch_med_imaging.pmi_data_loader import *
from pathlib import Path


class TestTorchIO(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTorchIO, self).__init__(*args, **kwargs)

    def tearDown(self) -> None:
        self.f.close()

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
                - [256, 256, 256]
                - padding_mode: 0
                - crop_pad_sides: cct    
            """
        self.f = tempfile.NamedTemporaryFile('w+', suffix='yaml')
        self.f.write(self.transform_1)
        self.f.flush()
        self.augment = create_transform_compose(self.f.name)
        self.sample = tio.Subject(image=tio.ScalarImage('./sample_data/img/MRI_01.nii.gz'))
        pass

    def _define_image_data_loader(self):
        data_loader_cfg = PMIImageDataLoaderCFG(
            input_dir     = './sample_data/img/',
            probmap_dir   = './sample_data/seg/',
            target_dir    = './sample_data/seg/',
            augmentation  = self.f.name,
            id_globber    = "^MRI_[0-9]+",
            sampler       = 'uniform', # Unset sampler to load the whole image
            sampler_kwargs    = dict(
                patch_size = [64, 64, 64]
            ),
            tio_queue_kwargs = dict(            # dict passed to ``tio.Queue``
                max_length             = 15,
                samples_per_volume     = 1,
                num_workers            = 8,
                shuffle_subjects       = True,
                shuffle_patches        = True,
                start_background       = True,
                verbose                = True,
            )
        )
        data_loader = PMIImageDataLoader(data_loader_cfg)
        return data_loader

    def _define_imagefeatpair_data_loader(self):
        data_loader_cfg = PMIImageFeaturePairLoaderCFG(
            input_dir     = './sample_data/img/',
            probmap_dir   = './sample_data/seg/',
            target_dir    = './sample_data/sample_class_gt.csv',
            target_column = 'Class',
            augmentation  = self.f.name,
            id_globber    = "^MRI_[0-9]+",
            sampler       = 'uniform', # Unset sampler to load the whole image
            sampler_kwargs    = dict(
                patch_size = [128, 128, 128]
            ),
            tio_queue_kwargs = dict(            # dict passed to ``tio.Queue``
                max_length             = 15,
                samples_per_volume     = 1,
                num_workers            = 8,
                shuffle_subjects       = True,
                shuffle_patches        = True,
                start_background       = True,
                verbose                = True,
            )
        )
        data_loader = PMIImageFeaturePairLoader(data_loader_cfg)
        return data_loader

    def test_transform(self):
        transformed = self.augment.apply_transform(self.sample)
        print(transformed)

    def test_weighted_sampler(self):
        data_loader = self._define_image_data_loader()
        dl = data_loader.get_torch_data_loader(3, exclude_augment=False)
        for mb in dl:
            self.assertTupleEqual(tuple(mb['input'][tio.DATA].shape),
                                  tuple([3, 1] + data_loader.sampler_kwargs['patch_size']))

    def test_imgfeat_sampler(self):
        data_loader = self._define_imagefeatpair_data_loader()
        dl = data_loader.get_torch_data_loader(3, exclude_augment=False)
        for mb in dl:
            self.assertTupleEqual(tuple(mb['input'][tio.DATA].shape),
                                  tuple([3, 1] + data_loader.sampler_kwargs['patch_size']))