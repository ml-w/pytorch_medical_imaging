import configparser
import inspect
import os
import re
import tempfile
import unittest
import torch
import copy
import pytest
from pathlib import Path

from mnts.mnts_logger import MNTSLogger
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.distributed as dist

from pytorch_med_imaging.controller import PMIController, PMIControllerCFG
from pytorch_med_imaging.lr_scheduler import *
from pytorch_med_imaging.solvers.earlystop import *

from sample_data.config.sample_controller_cfg import *

class TestController(unittest.TestCase):
    def __init__(self, *args, sample_config = "./sample_data/config/sample_config_seg.ini", **kwargs):
        self.sample_config = Path(sample_config)
        super(TestController, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        MNTSLogger('.',logger_name='unittest', verbose=True, keep_file=False, log_level='debug')
        cls._logger = MNTSLogger['unittest']
        cls._logger.debug(f"Class logger setup for {cls.__name__}")
        cls._checkpoint_dir = tempfile.TemporaryDirectory()
        cls._checkpoint_path = Path(cls._checkpoint_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls._checkpoint_dir.cleanup()

    def setUp(self):
        if self.__class__.__name__ == 'TestController':
            raise unittest.SkipTest("Base class.")

        # create temp output_dir and temp_config file
        self.temp_output_dir = tempfile.TemporaryDirectory()
        self.temp_output_path = Path(self.temp_output_dir.name)
        self.temp_config = tempfile.NamedTemporaryFile("a", suffix='.ini')
        self.temp_config_path = Path(self.temp_config.name)

        # prepare configurations
        self._prepare_cfg()

        # some temp dir items
        self.cfg.cp_save_dir = str(self._checkpoint_path.joinpath('sample_cp.pt'))
        self.cfg.cp_load_dir = str(self._checkpoint_path.joinpath('sample_cp.pt'))
        self.cfg.output_dir = str(self._checkpoint_path)

        self._prepare_controller()

    def override_config(self):
        return

    def tearDown(self):
        self.temp_output_dir.cleanup()
        self.temp_config.close()

    @abstractmethod
    def _preapre_cfg(self):
        r"""Prepare the cfg instance for the controller.
        """
        raise NotImplementedError

    def _prepare_controller(self):
        r"""Use the prepared controller to create the controller instance. Store the controller instance to
        :attr:`controller`."""
        self.controller = PMIController(self.cfg)

    def test_s1_create(self):
        new_cfg = copy.deepcopy(self.cfg)
        pass

    def test_s2_id_override(self):
        self.controller._pre_process_flags()

        # check if flags are correctly override
        override_config = configparser.ConfigParser()
        override_config.read('sample_data/config/sample_id_setting.ini')
        override_training_id = override_config['FileList']['training'].split(',')
        override_testing_id = override_config['FileList']['testing'].split(',')

        self.assertTupleEqual(tuple(self.controller.data_loader_cfg.id_list),
                              tuple(override_training_id))

    def test_s3_train(self):
        self.controller.train()

    def test_s4_inference(self):
        dummy_cp_path = self.temp_output_path.joinpath('temp_file')
        dummy_cp = torch.save(self.controller.solver_cfg.net.state_dict(),
                              str(dummy_cp_path))
        self.controller.solver_cfg.cp_load_dir = str(dummy_cp_path)
        self.controller.inference()

    def test_s5_kfold(self):
        self.controller.cfg.id_list = 'sample_data/config/sample_3_fold/{fold_code}.ini'
        self.controller.cp_save_dir = str(self._checkpoint_path.joinpath('checkpoint_{fold_code}.pt'))
        self.controller.cp_load_dir = str(self._checkpoint_path.joinpath('checkpoint_{fold_code}.pt'))
        self.controller.output_dir = str(self.temp_output_path.joinpath('output_{fold_code}'))

        # normally called during exec() but we are calling train/inference() directly, so call _pre_process_flags()
        # manually first to override 'fold_code' tags
        self.controller._pre_process_flags()

        # check the checkpoint is saved correctly
        self.controller.train()
        self.assertTrue(self._checkpoint_path.joinpath(f'checkpoint_{self.controller.cfg.fold_code}.pt').is_file())
        
        # check the inference output is written to correct place
        self.controller.inference()
        inf_output_dir = self.temp_output_path.joinpath(f'output_{self.controller.cfg.fold_code}')
        self.assertTrue(inf_output_dir.is_dir())
        self.assertGreater(len(list(inf_output_dir.iterdir())), 0, "No output found!")

    def test_s6_override(self):
        import yaml
        test_override_yaml = 'sample_data/config/sample_override/sample_override_setting_1.yaml'
        self.controller.override_cfg(test_override_yaml)

        with open(test_override_yaml, 'r') as f:
            _override = yaml.safe_load(f)
        if 'solver_cfg' in _override:
            for key, v in _override['solver_cfg'].items():
                msg = f"Expect key {key} overrided to be {v} but got {getattr(self.controller.solver_cfg, key)}."
                self.assertEqual(getattr(self.controller.solver_cfg, key), v, msg)

    def test_s7_ddptrain(self):
        world_size = torch.cuda.device_count()
        torch.multiprocessing.spawn(self.__class__.ddp_helper, args=(world_size, copy.deepcopy(self.cfg)),
                                    nprocs=world_size)

    @classmethod
    def ddp_helper(cls, rank, world_size, cfg):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "23455"
        dist.init_process_group("nccl", world_size=world_size, rank=rank)
        controller = PMIController(copy.deepcopy(cfg))
        controller.train()
        dist.barrier()
        dist.destroy_process_group()

class TestSegController(TestController):
    cls_cfg = SampleSegControllerCFG()
    def setUp(self):
        super(TestSegController, self).setUp()
        self.__class__.cls_cfg = self.cfg

    def _prepare_cfg(self):
        self.cfg = SampleSegControllerCFG()
