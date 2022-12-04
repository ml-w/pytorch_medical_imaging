import unittest
from pytorch_med_imaging.PMI_data_loader.pmi_dataloader_base import *
from pytorch_med_imaging.PMI_data_loader import *
from mnts.mnts_logger import MNTSLogger

class TestDataLoader(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDataLoader, self).__init__(*args, **kwargs)
        self.loader: PMIDataLoaderBase = None

    def test_load_training_data(self):
        loader = self.loader._load_data_set_training()
        for l in loader:
            print(l)
            break


    def test_load_testing_data(self):
        loader = self.loader._load_data_set_inference()
        for l in loader:
            print(l)
            break

class TestImageDataLoader(TestDataLoader):
    def setUpClass() -> None:
        MNTSLogger('.', logger_name='unittest', log_level='debug', keep_file=False)

    def tearDownClass() -> None:
        MNTSLogger.cleanup()

    def setUp(self):
        self.cfg = PMIImageDataLoaderCFG
        self.cfg.input_dir   = './sample_data/img/'
        self.cfg.target_dir  = './sample_data/seg/'
        self.cfg.mask_dir    = './sample_data/seg/'
        self.cfg.probmap_dir = './sample_data/seg/'
        self.cfg.id_globber  = "^\w+_\d+"
        self.cfg.sampler     = 'weighted'
        self.cfg.sampler_kwargs['patch_size'] = [3, 32, 32]
        self.loader = PMIImageDataLoader(self.cfg)
        pass
