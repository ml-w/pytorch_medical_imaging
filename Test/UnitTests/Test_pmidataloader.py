import unittest
from pytorch_med_imaging.PMI_data_loader.pmi_dataloader_base import *
from pytorch_med_imaging.PMI_data_loader import *
from mnts.mnts_logger import MNTSLogger

class TestDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        if self.__class__.__name__ == 'TestDataLoader':
            raise unittest.SkipTest("Base class.")
        self._logger = MNTSLogger('.', logger_name=self.__class__.__name__, log_level='debug', keep_file=False)

    def __init__(self, *args, **kwargs):
        super(TestDataLoader, self).__init__(*args, **kwargs)
        self.loader: PMIDataLoaderBase = None

    def test_load_training_data(self):
        loader = self.loader._load_data_set_training()
        self.assertEqual(len(loader), self.expected_training_queue_len)
        for l in loader:
            break
        return loader

    def test_load_inference_data(self):
        loader = self.loader._load_data_set_inference()
        self.assertEqual(len(loader), self.expected_inference_queue_len)
        for l in loader:
            break
        return loader


class TestImageDataLoader(TestDataLoader):
    def setUp(self):
        super(TestImageDataLoader, self).setUp()
        self.cfg = PMIImageDataLoaderCFG
        self.cfg.input_dir    = './sample_data/img/'
        self.cfg.target_dir   = './sample_data/seg/'
        self.cfg.mask_dir     = './sample_data/seg/'
        self.cfg.probmap_dir  = './sample_data/seg/'
        self.cfg.augmentation = './sample_data/config/sample_transform.yaml'
        self.cfg.id_globber   = "^\w+_\d+"
        self.cfg.id_list     = ['MRI_01', 'MRI_02']
        self.cfg.sampler     = 'weighted'
        self.cfg.sampler_kwargs['patch_size']           = [32, 32, 3]
        self.cfg.tio_queue_kwargs['samples_per_volume'] = 2
        self.cfg.inf_samples_per_vol                    = 5

        # expected variables
        self.num_subjects = len(self.cfg.id_list)
        self.expected_training_queue_len = self.num_subjects * self.cfg.tio_queue_kwargs['samples_per_volume']
        self.expected_inference_queue_len = self.num_subjects * self.cfg.inf_samples_per_vol

        # prepare loader
        self.loader = PMIImageDataLoader(self.cfg)
        pass

    def test_load_training_data(self):
        loader = super(TestImageDataLoader, self).test_load_training_data()
        for l in loader:
            self.assertTupleEqual(tuple(self.cfg.sampler_kwargs['patch_size']),
                                  tuple(l.shape[1:]))
            break

    def test_load_inference_data(self):
        loader = super(TestImageDataLoader, self).test_load_inference_data()
        for l in loader:
            self.assertTupleEqual(tuple(self.cfg.sampler_kwargs['patch_size']),
                                  tuple(l.shape[1:]))
            break

PMIImageFeaturePairLoader
class TestImageFeaturePairLoader(TestDataLoader):
    def setUp(self):
        super(TestImageFeaturePairLoader, self).setUp()
        self.cfg = PMIImageFeaturePairLoaderCFG
        self.cfg.input_dir   = './sample_data/img/'
        self.cfg.target_dir  = './sample_data/sample_binaryclass_gt.xlsx'
        self.cfg.mask_dir    = './sample_data/seg/'
        self.cfg.probmap_dir = './sample_data/seg/'
        self.cfg.id_globber  = "^\w+_\d+"
        self.cfg.id_list     = ['MRI_01', 'MRI_02']
        self.cfg.sampler     = 'weighted'
        self.cfg.sampler_kwargs['patch_size']           = [32, 32, 3]
        self.cfg.tio_queue_kwargs['samples_per_volume'] = 2
        self.cfg.inf_samples_per_vol                    = 5
        self.cfg.excel_sheetname = 'sample_binaryclass_gt'
        self.cfg.target_column   = 'Class'
        self.cfg.augmentation    = './sample_data/config/sample_transform.yaml'

        # expected variables
        self.num_subjects = len(self.cfg.id_list)
        self.expected_training_queue_len = self.num_subjects * self.cfg.tio_queue_kwargs['samples_per_volume']
        self.expected_inference_queue_len = self.num_subjects * self.cfg.inf_samples_per_vol

        # prepare loader
        self.loader = PMIImageFeaturePairLoader(self.cfg)
        pass

    def test_load_training_data(self):
        loader = super(TestImageFeaturePairLoader, self).test_load_training_data()
        for l in loader:
            msg = f"Expect integer class for ground-truth, got {type(l['gt'])} instead"
            self.assertIsInstance(l['gt'].item(), int, msg)
            self._logger.debug(f"{l}")
            break

    def test_load_inference_data(self):
        loader = super(TestImageFeaturePairLoader, self).test_load_inference_data()
        for l in loader:
            msg = f"Expect integer class for ground-truth, got {type(l['gt'])} instead"
            self.assertIsInstance(l['gt'].item(), int, msg)
            self._logger.debug(f"{l}")
            break


class TestImageFeaturePairLoaderConcat(TestImageFeaturePairLoader):
    def setUp(self):
        super(TestImageFeaturePairLoaderConcat, self).setUp()
        self.cfg.target_dir = "./sample_data/sample_concat_df.xlsx"
        self.cfg.excel_sheetname = None
        self.cfg.target_column = "conclusion"
        self.cfg.data_types = [float, str]

        self.loader = PMIImageFeaturePairLoaderConcat(self.cfg)

    def test_load_training_data(self):
        loader = super(TestImageFeaturePairLoader, self).test_load_training_data()
        for l in loader:
            msg = f"Expect integer class for ground-truth, got {type(l['gt'])} instead"
            self.assertIsInstance(l['gt'], str, msg)
            self._logger.debug(f"{l}")
            break

    def test_load_inference_data(self):
        loader = super(TestImageFeaturePairLoader, self).test_load_inference_data()
        for l in loader:
            msg = f"Expect integer class for ground-truth, got {type(l['gt'])} instead"
            self.assertIsInstance(l['gt'], str, msg)
            self._logger.debug(f"{l}")
            break


class TestPMIImageMCDataLoader(TestImageDataLoader):
    def setUp(self):
        super(TestPMIImageMCDataLoader, self).setUp()
        self.cfg = PMIImageMCDataLoaderCFG # note that super() already defined parent class attributes
        self.cfg.input_dir = './sample_data/'
        self.cfg.target_dir = './sample_data/'
        self.cfg.input_subdirs = ['img', 'img']
        self.cfg.target_subdirs = ['seg', 'seg']
        self.cfg.new_attr = ['img_new', 'seg_new']
        self.cfg.data_types = [float, 'uint8']
        self.loader = PMIImageMCDataLoader(self.cfg)

    def test_load_training_data(self):
        loader = super(TestImageDataLoader, self).test_load_training_data()
        for l in loader:
            _shape = l[self.cfg.new_attr[0]].shape
            self.assertTupleEqual(tuple(self.cfg.sampler_kwargs['patch_size']),
                                  tuple(_shape[1:]))
            self.assertEqual(len(self.cfg.input_subdirs),
                             _shape[0])
            break

    def test_load_inference_data(self):
        loader = super(TestImageDataLoader, self).test_load_inference_data()
        for l in loader:
            _shape = l[self.cfg.new_attr[0]].shape
            self.assertTupleEqual(tuple(self.cfg.sampler_kwargs['patch_size']),
                                  tuple(_shape[1:]))
            self.assertEqual(len(self.cfg.input_subdirs),
                             _shape[0])
            break