import tempfile
import unittest
import yaml
import copy
from abc import abstractmethod
from pytorch_med_imaging.solvers.earlystop import LossReferenceEarlyStop, BaseEarlyStop
from pytorch_med_imaging.lr_scheduler import PMILRScheduler
from sample_data.config.sample_cfg import *

from mnts.mnts_logger import MNTSLogger

from sample_data.config.sample_cfg import *


class TestSolver(unittest.TestCase):
    r"""Unittest for solvers.

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
        if self.__class__.__name__ == 'TestSolver':
            self.skipTest('Base test class')

        self._prepare_cfg()
        self._prepare_loader()
        self._prepare_solver()
        self.solver_cp_save_path = tempfile.NamedTemporaryFile('w', suffix='.pt')

    @abstractmethod
    def _prepare_cfg(self):
        r"""When implementing, you should create the following attributes:

        1. data_loader_cfg
        2. data_loader_cfg_cls
        3. solver_cfg
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_loader(self):
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
        new_cfg = copy.deepcopy(self.solver_cfg)
        pass

    def test_s2_step(self):
        for mb in self.data_loader.get_torch_data_loader(self.solver_cfg.batch_size):
            s, g = self.solver._unpack_minibatch(mb, self.solver_cfg.unpack_key_forward)
            self.solver.step(s, g)

            self.assertEqual(s.shape[0], self.solver_cfg.batch_size)
            if hasattr(self.data_loader_cfg, 'sampler_kwargs') and self.data_loader_cfg.sampler is not None:
                self.assertTupleEqual(tuple(s.shape[2:]),
                                      tuple(self.data_loader_cfg.sampler_kwargs['patch_size']))
            break
        pass

    def tearDown(self):
        self.solver_cp_save_path.close()

    def test_s3_fit(self):
        self.solver.set_data_loader(self.data_loader)
        self.solver.fit(self.solver_cp_save_path.name)

    def test_s4_early_stop(self):
        self._add_early_stop()
        self.solver.set_data_loader(self.data_loader)
        self.solver.fit(self.solver_cp_save_path.name)
        pass

    def _add_early_stop(self):
        self.solver.early_stop = LossReferenceEarlyStop(1, 1)

    def test_s5_validation(self):
        data_loader_cfg_train = self.data_loader_cfg_cls(
            id_list = ['MRI_01', 'MRI_02']
        )
        data_loader_cfg_test = self.data_loader_cfg_cls(
            id_list = ['MRI_03', 'MRI_04']
        )
        dataloader_train = self.data_loader_cls(data_loader_cfg_train)
        dataloader_test = self.data_loader_cls(data_loader_cfg_test)

        self.solver.set_data_loader(dataloader_train, dataloader_test)
        self.solver.fit(self.solver_cp_save_path.name)

    def test_max_step(self):
        self.solver.set_data_loader(self.data_loader)
        self.solver.max_step = 1
        self.solver.fit(self.solver_cp_save_path.name)

class TestSegmentationSolver(TestSolver):
    def setUp(self):
        super(TestSegmentationSolver, self).setUp()

    def _prepare_cfg(self):
        self.data_loader_cfg = SampleSegLoaderCFG()
        self.data_loader_cfg_cls = SampleSegLoaderCFG
        self.solver_cfg = SampleSegSolverCFG()
        self.solver_cfg.debug = True

    def _prepare_loader(self):
        self.data_loader = PMIImageDataLoader(self.data_loader_cfg)
        self.data_loader_cls = PMIImageDataLoader

    def _prepare_solver(self):
        self.solver = SegmentationSolver(self.solver_cfg)
        self.solver_cls = SegmentationSolver


class TestClassificationSolver(TestSolver):
    def _prepare_cfg(self):
        self.data_loader_cfg = SampleClsLoaderCFG()
        self.data_loader_cfg_cls = SampleClsLoaderCFG
        self.solver_cfg = SampleClsSolverCFG()
        self.solver_cfg.debug = True

    def _prepare_loader(self):
        self.data_loader = PMIImageFeaturePairLoader(self.data_loader_cfg)
        self.data_loader_cls = PMIImageFeaturePairLoader

    def _prepare_solver(self):
        self.solver = ClassificationSolver(self.solver_cfg)
        self.solver_cls = ClassificationSolver

    def test_report_missclassification(self):
        dic = torch.Tensor([0, 1, 1, 0])
        g   = torch.Tensor([1, 1, 1, 0])
        uids = ['1', '2', '3' , '4']
        self.solver._update_misclassification_record(dic, g, uids)

class TestBinaryClassificationSolver(TestClassificationSolver):
    def _prepare_cfg(self):
        super(TestBinaryClassificationSolver, self)._prepare_cfg()
        self.data_loader_cfg.target_dir = './sample_data/sample_binaryclass_gt.xlsx'
        self.data_loader_cfg_cls.target_dir = './sample_data/sample_binaryclass_gt.xlsx'
        self.solver_cfg = SampleBinClsSolverCFG()
        self.solver_cfg.debug = True

    def _prepare_solver(self):
        self.solver = BinaryClassificationSolver(self.solver_cfg)
        self.solver_cls = BinaryClassificationSolver

class TestSolverCreateFromFlags(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._logger = MNTSLogger(".", logger_name='unittest', keep_file=False, log_level='debug', verbose=True)

    def setUp(self) -> None:
        with open('./sample_data/config/sample_override/sample_override_setting_2.yaml', 'r') as f:
            self._override_settings = yaml.safe_load(f)

        # Use segmentation
        self.data_loader_cfg = SampleSegLoaderCFG()
        self.data_loader_cfg_cls = SampleSegLoaderCFG
        self.solver_cfg = SampleSegSolverCFG()
        self.solver_cfg.debug = True
        self.data_loader = PMIImageDataLoader(self.data_loader_cfg)
        self.data_loader_cls = PMIImageDataLoader

    def test_create_lr_sche(self):
        for k, v in self._override_settings['test_solver_cfg_1'].items():
            setattr(self.solver_cfg, k, v)
        solver = SegmentationSolver(self.solver_cfg)
        solver.prepare_lr_scheduler()
        self.assertIsInstance(solver.lr_sche, PMILRScheduler)

    def test_create_early_stopper(self):
        for k, v in self._override_settings['test_solver_cfg_2'].items():
            setattr(self.solver_cfg, k, v)
        solver = SegmentationSolver(self.solver_cfg)
        self.assertIsInstance(solver.early_stop, BaseEarlyStop)
        pass
