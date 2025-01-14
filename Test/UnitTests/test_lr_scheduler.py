import configparser
import inspect
import os
import re
import tempfile
import unittest
import torch
from pathlib import Path

from mnts.mnts_logger import MNTSLogger
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from pytorch_med_imaging.controller.pmi_controller import PMIController
from pytorch_med_imaging.lr_scheduler import *
from sample_data.config.sample_controller_cfg import SampleClsControllerCFG
from sample_data.config.sample_cfg import SampleClsLoaderCFG
import copy

class TestLRSchedulerWithController(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # define cls controller cfg for testing all LR schedulers
        self.sample_config = SampleClsControllerCFG()
        super(TestLRSchedulerWithController, self).__init__(*args, **kwargs)

    def setUp(self):
        # create temp output_dir and temp_config file
        self.temp_output_dir = tempfile.TemporaryDirectory()
        self.temp_output_path = Path(self.temp_output_dir.name)

        # create the controller
        self.ctl_cfg = copy.copy(self.sample_config)
        self.ctl_cfg.output_dir = str(self.temp_output_path)
        self.ctl_cfg.cp_save_dir = str(self.temp_output_path.joinpath("temp_cp.pt"))
        self.ctl_cfg.debug_mode = True

    def test_onecyclelr_losses(self):
        # One cycle LR must be tested this way
        self.ctl_cfg.solver_cfg.lr_sche = "OneCycleLR"
        self.ctl_cfg.solver_cfg.lr_sche_args = [1E-4] # max_lr
        self.ctl_cfg.solver_cfg.lr_sche_kwargs = {'total_steps':50,'cycle_momentum':True}
        self.ctl_cfg.solver_cfg.num_of_epochs = 3
        self.ctl_cfg.solver_cfg.optimizer = 'SGD'
        self.ctl_cfg.solver_cfg.init_mom = 0.99
        controller = PMIController(self.ctl_cfg)
        controller.train()


class TestPMILRScheduler(unittest.TestCase):
    def __init__(self, *args):
        super(TestPMILRScheduler, self).__init__(*args)
        net = torch.nn.Linear(10, 2).train()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=1E-5)
        self.name = None
        self.args = []
        self.kwargs = {}

    def setUp(self):
        if self.__class__.__name__ == 'TestPMILRScheduler':
            raise unittest.SkipTest("Base class.")

        self._logger = MNTSLogger(self.__class__.__name__, logger_name='unittest', verbose=True, keep_file=False,
                                  log_level='debug')
        self._logger.debug(f"args: {self.args}")
        self._logger.debug(f"kwargs: {self.kwargs}")
        PMILRScheduler.reset()
        PMILRScheduler(self.name, *self.args, **self.kwargs)

    # def tearDown(self):
    #     PMILRScheduler.hard_reset()

    def test_create_instance(self):
        PMILRScheduler.set_optimizer(self.optimizer)

    def test_step(self):
        PMILRScheduler.set_optimizer(self.optimizer)
        start_lr = PMILRScheduler.get_last_lr()
        PMILRScheduler.step_scheduler()
        end_lr = PMILRScheduler.get_last_lr()
        print(f"before: {end_lr}, after: {start_lr}")

class TestExponentialLR(TestPMILRScheduler):
    def setUp(self):
        self.name = 'ExponentialLR'
        self.args = [0.9]
        self.kwargs = {}
        super(TestExponentialLR, self).setUp()


class TestStepLR(TestPMILRScheduler):
    def setUp(self):
        self.name = 'StepLR'
        self.args = [10] # step_size
        self.kwargs = dict(gamma=0.5)
        super(TestStepLR, self).setUp()

class TestReduceLROnPlateau(TestPMILRScheduler):
    def setUp(self):
        self.name = 'ReduceLROnPlateau'
        self.args = list()
        self.kwargs = dict(mode='min', factor=0.8, patience=20, threshold=1E-5)
        super(TestReduceLROnPlateau, self).setUp()

    def test_step(self):
        PMILRScheduler.set_optimizer(self.optimizer)
        PMILRScheduler.step_scheduler(1E-4)
        print(PMILRScheduler.get_last_lr())


class TestMultiStepLR(TestPMILRScheduler):
    def setUp(self):
        self.name = 'MultiStepLR'
        self.args = [
            [30, 80] # milestone
        ]
        super(TestMultiStepLR, self).setUp()


class TestLambdaLR(TestPMILRScheduler):
    def setUp(self):
        self.name = 'LambdaLR'
        self.args = [
            lambda x: x * 0.95 # lambda
        ]
        super(TestLambdaLR, self).setUp()


class TestCosineAnnealingLR(TestPMILRScheduler):
    def setUp(self):
        self.name = 'CosineAnnealingLR'
        self.args = [
            10 # T_max
        ]
        super(TestCosineAnnealingLR, self).setUp()


class TestConstantLR(TestPMILRScheduler):
    def setUp(self):
        self.name = 'ConstantLR'
        super(TestConstantLR, self).setUp()


class TestLinearLR(TestPMILRScheduler):
    def setUp(self):
        self.name = 'LinearLR'
        super(TestLinearLR, self).setUp()

class TestDecayCAWR(TestPMILRScheduler):
    def setUp(self):
        self.name = 'DecayCAWR'
        self.args = [
            0.99, # exp_factor
            10  , # T_0
        ]
        self.kwargs = dict(
            T_mult=2,
            eta_min = 1E-6
        )
        super(TestDecayCAWR, self).setUp()

class TestDecayCAWR_n_EXP(TestPMILRScheduler):
    def setUp(self):
        self.name = 'DecayCAWR_n_EXP'
        self.args = [
            0.99, # exp_factor_dawr
            0.95, # gamma
            10  , # T_0
            50  , # T_cut
        ]
        self.kwargs = dict(
            T_mult=2,
            eta_min = 1E-6
        )
        super(TestDecayCAWR_n_EXP, self).setUp()

class TestOneCycleLR(TestPMILRScheduler):
    def setUp(self):
        self.name = 'OneCycleLR'
        self.args = [1E-2] # max_lr
        self.kwargs = dict(
            steps_per_epoch=2,
            epochs= 3,
            base_momentum = 0.5,
        )
        super(TestOneCycleLR, self).setUp()