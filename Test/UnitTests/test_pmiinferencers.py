import tempfile
import unittest
import pytest
from sample_data.config.sample_cfg import *

from mnts.mnts_logger import MNTSLogger
from pytorch_med_imaging.inferencers import *
from pytorch_med_imaging.pmi_data_loader import PMIDataLoaderBase, PMIDataLoaderBaseCFG


class TestInferencer(unittest.TestCase):
    r"""Unittest for inferencers.

    Attributes:
        data_loader_cls (type):
            The cfg class of the data loader. E.g. :class:`SampleSegLoaderCFG`
        data_loader_cfg (PMIDataLoaderBaseCFG):
            The cfg instance for creating loader.
        data_loader (PMIDataLoaderBase):
            The instance of data loader for training.
        data_loader_val (PMIDataLoaderBase):
            The instance of data loader for validation.
        solver_cls (type):
            Type of solver invovled.
        solver_cfg (type):
            Class of the solver cfg.
        solver (SolverBase):
            Solver instance created using the upper cfg.
    """
    @classmethod
    def setUpClass(cls):
        cls._logger = MNTSLogger(".", logger_name='unittest', keep_file=False, log_level='debug', verbose=True)

    def setUp(self):
        if self.__class__.__name__ == 'TestInferencer':
            self.skipTest('Base test class')

        # create a temp folder to host the outputs
        self.temp_output_dir = tempfile.TemporaryDirectory()

        self._prepare_cfg()
        self._prepare_loader()

        # Save a dummy checkpoint to tempfile first
        self.cp_load_tmp_file = tempfile.NamedTemporaryFile(suffix='.pt', mode='w+')
        self.cp_load_dir = self.cp_load_tmp_file.name
        torch.save(self.inferencer_cfg.net.state_dict(), self.cp_load_dir)
        self.inferencer_cfg.cp_load_dir = self.cp_load_dir

        self._prepare_inferencer()


    def tearDown(self):
        self.cp_load_tmp_file.close()
        self.temp_output_dir.cleanup()

    @abstractmethod
    def _prepare_cfg(self):
        r"""When implementing, you should create the following attributes:

        1. data_loader_cfg
        2. data_loader_cfg_cls
        3. solver_cfg
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_inferencer(self):
        r"""When implementing, you should create the following attributes:

        1. data_loader
        2. data_loader_cls
        3. data_loader_val
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_solver(self):
        r"""When implementing, you should create the following attributes:

        1. solver
        2. solver_cls
        """
        raise NotImplementedError

    def test_s1_create(self):
        pass

    def test_s2_write_out(self):
        self.inferencer.set_data_loader(self.data_loader)
        self.inferencer.write_out()
        self.inferencer.display_summary()

    def test_set_data_loader(self):
        self.inferencer.set_data_loader(self.data_loader)
        assert self.inferencer.data_loader == self.data_loader

    def test_load_checkpoint_nonexistent_path(self):
        with pytest.raises(IOError):
            self.inferencer.load_checkpoint('nonexistent_path')


class TestSegmentationInferencer(TestInferencer):
    def _prepare_cfg(self):
        self.data_loader_cfg = SampleSegLoaderCFG(run_mode='inference')
        self.data_loader_cfg_cls = SampleSegLoaderCFG
        self.inferencer_cfg = SampleSegSolverCFG
        self.inferencer_cfg.output_dir = self.temp_output_dir.name
        self.inferencer_cfg.debug = True

    def _prepare_loader(self):
        self.data_loader = PMIImageDataLoader(self.data_loader_cfg)
        self.data_loader_cls = PMIImageDataLoader

    def _prepare_inferencer(self):
        self.inferencer = SegmentationInferencer(self.inferencer_cfg)
        self.inferencer_cls = SegmentationInferencer
        self.inferencer.load_checkpoint()

class TestClassificationInferencer(TestInferencer):
    def _prepare_cfg(self):
        self.data_loader_cfg = SampleClsLoaderCFG(run_mode='inference')
        self.data_loader_cfg_cls = SampleClsLoaderCFG
        self.inferencer_cfg = SampleClsSolverCFG(output_dir = self.temp_output_dir.name,
                                                 debug = True)

    def _prepare_loader(self):
        self.data_loader = PMIImageFeaturePairLoader(self.data_loader_cfg)
        self.data_loader_cls = PMIImageFeaturePairLoader

    def _prepare_inferencer(self):
        self.inferencer = ClassificationInferencer(self.inferencer_cfg)
        self.inferencer_cls = ClassificationInferencer

class TestBinaryClassificationInferencer(TestClassificationInferencer):
    def _prepare_cfg(self):
        super(TestBinaryClassificationInferencer, self)._prepare_cfg()
        self.data_loader_cfg = SampleClsLoaderCFG(run_mode='inference')
        self.data_loader_cfg.target_dir = './sample_data/sample_binaryclass_gt.xlsx'
        self.data_loader_cfg_cls.target_dir = './sample_data/sample_binaryclass_gt.xlsx'
        self.inferencer_cfg = SampleBinClsSolverCFG(output_dir = self.temp_output_dir.name,
                                                    debug=True)

    def _prepare_inferencer(self):
        self.inferencer = BinaryClassificationInferencer(self.inferencer_cfg)
        self.inferencer_cls = BinaryClassificationInferencer
