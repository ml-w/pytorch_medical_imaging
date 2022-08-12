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

from pytorch_med_imaging.pmi_controller import PMIController
from pytorch_med_imaging.lr_scheduler import *

class TestLRScheduler(unittest.TestCase):
    def __init__(self, *args, sample_config = "./sample_data/config/sample_config_seg.ini", **kwargs):
        self.sample_config = Path(sample_config)
        super(TestLRScheduler, self).__init__(*args, **kwargs)

    def setUp(self):
        # create temp output_dir and temp_config file
        self.temp_output_dir = tempfile.TemporaryDirectory()
        self.temp_output_path = Path(self.temp_output_dir.name)
        self.temp_config = tempfile.NamedTemporaryFile("a", suffix='.ini')
        self.temp_config_path = Path(self.temp_config.name)

        # replace logger
        self._logger = MNTSLogger(self.temp_output_dir.name + "/log",
                                  logger_name='unittest', verbose=True, keep_file=False, log_level='debug')

        # create the controller
        config_obj = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config_obj.read(self.sample_config)
        self.config = config_obj
        self.config['Data']['output_dir'] = str(self.temp_output_path)
        self.config['Checkpoint']['cp_save_dir'] = str(self.temp_output_path.joinpath("temp_cp.pt"))


    def build_controller(self):
        self.config.write(self.temp_config)
        self.temp_config.flush()
        pass

    def test_onecyclelr_losses(self):
        self.config['SolverParams']['lr_scheduler'] = "OneCycleLR"
        self.config['SolverParams']['lr_scheduler_args'] = "[]"
        self.config['SolverParams']['lr_scheduler_kwargs'] = "{'max_lr':1E-4,'total_steps':50,'cycle_momentum':True}"
        self.config['SolverParams']['num_of_epochs'] = '3'
        self.build_controller()
        controller = PMIController(self.temp_config.name, a=None)
        solver = controller.create_solver(controller.general_run_type)
        loader, loader_val = controller.prepare_loaders()
        solver.set_dataloader(loader, loader_val)
        controller.create_lr_scheduler(solver)
        self.assertIsInstance(solver.lr_scheduler, lr_scheduler.OneCycleLR)


