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


class TestController(unittest.TestCase):
    def __init__(self, *args, sample_config = "./sample_data/config/sample_config_seg.ini", **kwargs):
        self.sample_config = Path(sample_config)
        super(TestController, self).__init__(*args, **kwargs)

    def setUp(self):
        # create temp output_dir and temp_config file
        self.temp_output_dir = tempfile.TemporaryDirectory()
        self.temp_output_path = Path(self.temp_output_dir.name)
        self.temp_config = tempfile.NamedTemporaryFile("a", suffix='.ini')
        self.temp_config_path = Path(self.temp_config.name)

        # replace logger
        self.logger = MNTSLogger(self.temp_output_dir.name + "/log",
                                 logger_name='unittest', verbose=True, keep_file=False, log_level='debug')

        # create the controller
        self.temp_config.writelines(self.sample_config.open('r').readlines())
        self.temp_config.flush()
        config_obj = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config_obj.read(self.temp_config.name)
        self.config = config_obj
        self.config['Data']['output_dir'] = str(self.temp_output_path)
        self.config['Checkpoint']['cp_save_dir'] = str(self.temp_output_path.joinpath("temp_cp.pt"))

        self.controller = PMIController(self.temp_config.name, a=None)

    def tearDown(self):
        self.temp_output_dir.cleanup()
        self.temp_config.close()

    def test_argparse_override(self):
        class dummy(object):
            def __init__(self):
                super(dummy, self).__init__()
                self.debug = True
                self.debug_validation = True
                self.train = True
                self.inference = True
                self.batch_size = 4
                self.epoch = 5
                self.lr = 1E-5
                self.override = "(Filters,re_suffix)=\w+;(RunParams,batch_size)=8"

            def __getattr__(self, item):
                if not item in self.__dict__:
                    return None

        a = dummy()
        self.controller.override_config(a)
        self.assertEqual(8, self.controller.runparams_batch_size)
        self.assertEqual("\w+", self.controller.filters_re_suffix)
        self.assertTrue(self.controller.general_debug)
        self.assertTrue(self.controller.general_debug_validation)

    def test_solver_create(self):
        #TODO: test all kind of solvers
        solver = self.controller.create_solver(self.controller.general_run_type)
        self.assertEqual(solver.__class__.__name__,
                         re.search("^[\w]+", self.controller.general_run_type).group() + "Solver")

    def test_solver_net_create(self):
        solver = self.controller.create_solver(self.controller.general_run_type)
        net_name = solver.net.__class__.__name__
        self.assertEqual(net_name, re.search("^[\w]+", self.controller.network_network_type).group())

    def test_create_tb_writter(self):
        from pytorch_med_imaging.tb_plotter import TB_plotter
        os.environ["TENSORBOARD_LOGIDR"] = str(self.temp_output_path.joinpath("TB_logdir"))
        self.controller.general_plot_tb = True
        self.controller.prepare_tensorboard_writter()
        self.assertIsInstance(self.controller.writer, TB_plotter)

    def test_pmi_load_data(self):
        loader, loader_val = self.controller.prepare_loaders()
        self.assertEqual(self.controller.pmi_data.__class__.__name__,
                         self.controller.loaderparams_pmi_datatype_name)
        self.assertIsInstance(loader, DataLoader)
        self.assertIsInstance(loader_val, DataLoader)

    def test_inferencer_create(self):
        # TODO: create inferencer requires checkpoint, need to put that into sample_data
        # infer = self.controller.create_inferencer(self.controller.run_type)
        # self.assertEqual(solver.__class__.__name__,
        #                  re.search("^[\w]+", self.controller.run_type).group() + "Inferencer")
        pass

    def test_unpack_config(self):
        checks = {
            ('RunParams', 'batch_size')          : int,
            ('SolverParams', 'learning_rate')    : float,
            ('SolverParams', 'momentum')         : float,
            ('SolverParams', 'num_of_epochs')    : int,
            ('SolverParams', 'decay_rate_LR')    : float,
            ('SolverParams', 'lr_scheduler_dict'): dict,
            ('SolverParams', 'early_stop')       : dict
        }
        for keys in checks:
            try:
                self.assertIsInstance(getattr(self.controller, '_'.join(keys).lower()), checks[keys])
            except:
                self.fail(f"Error when checking key: {keys}")

    def test_pack_config(self):
        packed = self.controller._pack_config('SolverParams')
        self.assertTrue(all([p.split('_')[0] == 'solverparams' for p in packed]))

        packed = self.controller._pack_config(['SolverParams', 'General'])
        self.assertTrue(all([p.split('_')[0] in ('solverparams', 'general') for p in packed]))

class TestSolvers(TestController):
    def __init__(self, *args, **kwargs):
        super(TestSolvers, self).__init__(*args, **kwargs)
        
    def setUp(self):
        super(TestSolvers, self).setUp()
        self.solver = self.controller.create_solver(self.controller.general_run_type)
        
    def test_create_lr_scheduler(self):
        scheduler_args = {
            'LambdaLR': [
                [lambda epoch: 0.95 ** epoch]
            ],
            'StepLR': [
                [0.1], # step size
            ],
            'ConstantLR': [
                [], # if no args, needs to put an empty list here
                {'total_iters': 5,
                 'factor': 0.5}
            ],
            'LinearLR': [
                [],
                {'start_factor': 0.5,
                 'end_factor': 1}
            ],
            'ExponentialLR': [
                [0.99], # gamma,
                {} # if no kwargs, put empty dict here
            ],
            'CosineAnnealingWarmRestarts': [
                [50],
                {'T_mult': 2,
                 'eta_min': 1E-6
                 }
            ]
        }
        for key, val in scheduler_args.items():
            _args = val[0]
            _kwargs = val[1] if len(val) > 1 else {}
            try:
                self.solver.set_lr_scheduler(key, *_args, **_kwargs)
                self.logger.info(f"Passed for {key}")
            except Exception as e:
                self.fail(f"Fail when creating lr_scheduler {key}")
        pass

    def test_create_lossfunction(self):
        self.solver.create_lossfunction()

    def test_create_optimizer(self):
        self.solver.create_optimizer(self.solver.net.parameters())

    def test_validation(self):
        self.solver._last_epoch_loss = 10
        self.solver._last_val_loss = 15
        self.solver.solverparams_num_of_epochs = 2
        loader, loader_val = self.controller.prepare_loaders()
        self.solver.set_dataloader(loader, loader_val)
        self.solver.fit(str(self.temp_output_path.joinpath("test.pt")),
                        True)
        self.assertTrue(len(list(self.temp_output_path.glob("*pt"))) != 0)

    def test_fit(self):
        self.solver._last_epoch_loss = 10
        self.solver._last_val_loss = 15
        self.solver.solverparams_num_of_epochs = 2
        loader, loader_val = self.controller.prepare_loaders()
        self.solver.set_dataloader(loader, loader_val)
        self.solver.set_lr_scheduler('LinearLR', *[], **{'start_factor': 0.5,'end_factor': 1})
        self.solver.fit(str(self.temp_output_path.joinpath("test.pt")),
                        False)
        self.assertTrue(len(list(self.temp_output_path.glob("*pt"))) != 0)

    def test_match_type_with_network(self):
        out = self.solver._match_type_with_network(torch.IntTensor([1, 2, 3, 4, 5]))
        self.assertEqual(out.type(), self.solver._net_weight_type)

    def test_decay_optimizer(self):
        self.solver.set_lr_scheduler('LinearLR', *[], **{'start_factor': 0.5,'end_factor': 1})
        before = self.solver.get_last_lr()
        self.solver.decay_optimizer()
        after = self.solver.get_last_lr()
        self.assertLess(before, after)


class TestSegmentationSolver(TestSolvers):
    def __init__(self, *args, **kwargs):
        super(TestSegmentationSolver, self).__init__(*args, **kwargs)

class TestClassificaitonSolver(TestSolvers):
    def __init__(self, *args, **kwargs):
        super(TestClassificaitonSolver, self).__init__(
            *args,
            sample_config = "./sample_data/config/sample_config_class.ini",
            **kwargs)

class TestBinaryClassificaitonSolver(TestSolvers):
    def __init__(self, *args, **kwargs):
        super(TestBinaryClassificaitonSolver, self).__init__(
            *args,
            sample_config = "./sample_data/config/sample_config_binaryclass.ini",
            **kwargs)

class TestrAIdiologistSolver(TestSolvers):
    def __init__(self, *args, **kwargs):
        super(TestrAIdiologistSolver, self).__init__(
            *args,
            sample_config = "./sample_data/config/sample_config_rAIdiologist.ini",
            **kwargs)