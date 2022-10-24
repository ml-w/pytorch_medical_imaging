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

class TestController(unittest.TestCase):
    def __init__(self, *args, sample_config = "./sample_data/config/sample_config_seg.ini", **kwargs):
        self.sample_config = Path(sample_config)
        super(TestController, self).__init__(*args, **kwargs)

    def setUp(self):
        if self.__class__.__name__ == 'TestController':
            raise unittest.SkipTest("Base class.")

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
        self.override_config()
        self.config.write(self.temp_config)
        # self.temp_config.writelines(self.sample_config.open('r').readlines())
        self.temp_config.flush()

        self.controller = PMIController(self.temp_config.name, a=None)

    def override_config(self):
        return

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
        if not self.__class__.__name__.find('Solver') > -1:
            raise unittest.SkipTest("Skip for not testing solvers")
        solver = self.controller.create_solver(self.controller.general_run_type)
        self.assertEqual(solver.__class__.__name__,
                         re.search("^[\w]+", self.controller.general_run_type).group() + "Solver")

    def test_solver_net_create(self):
        if not self.__class__.__name__.find('Solver') > -1:
            raise unittest.SkipTest("Skip for not testing solvers")
        solver = self.controller.create_solver(self.controller.general_run_type)
        net_name = solver.net.__class__.__name__
        self.assertEqual(net_name, re.search("^[\w]+", self.controller.network_network_type).group())

    def test_inferencer_create(self):
        if not self.__class__.__name__.find('Inferencer') > -1:
            raise unittest.SkipTest("Skip for not testing inferencer")
        infer = self.controller.create_inferencer(self.controller.general_run_type)
        self.assertEqual(infer.__class__.__name__,
                         re.search("^[\w]+", self.controller.general_run_type).group() + "Inferencer")

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
        self._logger.debug("Loader create passed.")

        # Test unpack minibatch
        solver = self.controller.create_solver(self.controller.general_run_type)
        for i, mb in enumerate(loader):
            row = solver._unpack_minibatch(mb, [('input', 'gt'), 'gt'])
            self.assertIsInstance(row[0], tuple)
            self.assertIsInstance(row[1], torch.Tensor)
            self.assertEqual(2, len(row[0]))
            break
        self._logger.debug("Unpack minibatch passed.")

    def test_unpack_config(self):
        checks = {
            ('RunParams', 'batch_size')          : int,
            ('SolverParams', 'learning_rate')    : float,
            ('SolverParams', 'momentum')         : float,
            ('SolverParams', 'num_of_epochs')    : int,
            ('SolverParams', 'decay_rate_LR')    : float,
            ('SolverParams', 'lr_scheduler_dict'): (dict, None),
            ('SolverParams', 'early_stop')       : (dict, None)
        }
        for keys in checks:
            try:
                if getattr(self.controller, self.controller._make_dict_key(*keys)) is None:
                    continue
                self.assertIsInstance(getattr(self.controller, self.controller._make_dict_key(*keys)), checks[keys])
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
        if self.__class__.__name__ == 'TestSolvers':
            raise unittest.SkipTest("Base class.")
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
            ],
            'DecayCAWR': [
                [0.95, 10], # exp, T0
                {'T_mult': 1,
                 'eta_min': 1E-6}
            ]
        }
        self._logger.info("Testing creating of learning rate schedulers.")
        for key, val in scheduler_args.items():
            _args = val[0]
            _kwargs = val[1] if len(val) > 1 else {}
            try:
                self.solver.set_lr_scheduler(key, *_args, **_kwargs)
                self._logger.info(f"Passed for {key}.")
            except Exception as e:
                self.fail(f"Fail when creating lr_scheduler {key}")

    def test_create_lr_schedulers_from_config(self):
        scheduler_args = {
            # Lambda is unavailble duel to the nature of ast.literal_eval
            # 'LambdaLR': [
            #     "[lambda epoch: 0.95 ** epoch]",
            #     "{}"
            # ],
            'StepLR': [
                "[0.1]", # step size
                "{}"
            ],
            'ConstantLR': [
                "[]", # if no args, needs to put an empty list here
                "{'total_iters': 5,'factor': 0.5}"
            ],
            'LinearLR': [
                "[]",
                "{'start_factor': 0.5,'end_factor': 1}"
            ],
            'ExponentialLR': [
                "[0.99]", # gamma,
                "{}" # if no kwargs, put empty dict here
            ],
            'CosineAnnealingWarmRestarts': [
                "[50]",
                "{'T_mult': 2,'eta_min': 1E-6}"
            ],
            'DecayCAWR': [
                "[0.95, 10]", # exp, T0
                "{'T_mult': 1,'eta_min': 1E-6}"
            ]
        }
        # Test create from config
        self._logger.info("Testing creating of learning scheduler using config file.")
        for key, val in scheduler_args.items():
            self.config['SolverParams']['lr_scheduler'] = key
            self.config['SolverParams']['lr_scheduler_args'] = val[0]
            self.config['SolverParams']['lr_scheduler_kwargs'] = val[1]
            try:
                self.controller._unpack_config(self.config)
                if isinstance(self.controller.solverparams_lr_scheduler_args, (float, int, str)):
                    self.controller.solverparams_lr_scheduler_args = [self.controller.solverparams_lr_scheduler_args]
                self.solver.set_lr_scheduler(self.controller.solverparams_lr_scheduler,
                                             *self.controller.solverparams_lr_scheduler_args,
                                             **self.controller.solverparams_lr_scheduler_kwargs)
                self._logger.info(f"Passed for {key}.")
            except:
                self.fail(f"Fail when creating lr_schedule {key} from config.")

    def test_create_lossfunction(self):
        self.solver.create_lossfunction()

    def test_create_optimizer(self):
        self.solver.create_optimizer(self.solver.net.parameters())
        self.assertEqual(self.solver.solverparams_learning_rate,  self.solver.get_last_lr())

    def test_validation(self):
        self.solver._last_epoch_loss = 10
        self.solver._last_val_loss = 15
        self.solver.solverparams_num_of_epochs = 2
        loader, loader_val = self.controller.prepare_loaders()
        self.solver.set_dataloader(loader, loader_val)
        self.solver.fit(str(self.temp_output_path.joinpath("test.pt")),
                        True)
        self.assertTrue(len(list(self.temp_output_path.glob("*pt"))) != 0)

    def test_early_stop(self):
        from pytorch_med_imaging.solvers.SolverBase import SolverEarlyStopScheduler
        early_stop = {'method'  : 'LossReference', 'warmup': 0, 'patience': 2}
        self.solver._early_stop_scheduler = SolverEarlyStopScheduler(early_stop)
        self.solver.solverparams_num_of_epochs= 15
        self.controller.runparams_batch_size = 2
        loader, loader_val = self.controller.prepare_loaders()
        self.solver.set_dataloader(loader, loader_val)
        self.solver.fit(str(self.temp_output_path.joinpath("test.pt")),
                        False)
        self.assertTrue(self.solver._early_stop_scheduler._last_epoch < 14)

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

    def test_accumulate_grad(self):
        # manually set params
        accumulate_grad = 4
        self.solver.solverparams_accumulate_grad = accumulate_grad
        self.controller.runparams_batch_size = 2
        loader, loader_val = self.controller.prepare_loaders()
        self.solver.set_dataloader(loader, loader_val)
        for step_idx, mb in enumerate(self.solver._data_loader):
            s, g = self.solver._unpack_minibatch(mb, self.solver.solverparams_unpack_keys_forward)
            msg = f"Error at {step_idx}"
            self.assertEqual(step_idx % self.solver.solverparams_accumulate_grad,
                             self.solver._accumulated_steps,
                             msg)
            out, loss = self.solver.step(s, g)
            del s, g, mb

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


class TestInferencer(TestController):
    def __init__(self, *args, **kwargs):
        super(TestInferencer, self).__init__(*args, **kwargs)

    def setUp(self):
        if self.__class__.__name__ == 'TestInferencer':
            raise unittest.SkipTest("Base class.")
        super(TestInferencer, self).setUp()
        self.infer = self.controller.create_inferencer(self.controller.general_run_type)

    def override_config(self, override_config ={}):
        override_dict = {
            ('General', 'run_mode'): 'inference'
        }
        override_dict.update(override_config)
        for (section, key), value in override_dict.items():
            self.config[section][key] = str(value)

    def test_inference(self):
        self._logger.info(f"Crearing checkpoint: {self.controller.checkpoint_cp_load_dir}")
        with tempfile.NamedTemporaryFile(suffix=".pt", mode='w+') as tmp_checkpoint:
            self.controller.checkpoint_cp_load_dir = tmp_checkpoint.name
            torch.save(self.controller.net.state_dict(), tmp_checkpoint.name)
            self.controller.run()

    def test_inference_no_gt(self):
        self._logger.info(f"Crearing checkpoint: {self.controller.checkpoint_cp_load_dir}")
        with tempfile.NamedTemporaryFile(suffix=".pt", mode='w+') as tmp_checkpoint:
            self.controller.pmi_data._target_dir = None
            self.controller.checkpoint_cp_load_dir = tmp_checkpoint.name
            torch.save(self.controller.net.state_dict(), tmp_checkpoint.name)
            self.controller.run()


class TestSegmentationInferencer(TestInferencer):
    def __init__(self, *args, **kwargs):
        super(TestSegmentationInferencer, self).__init__(
            *args,
            sample_config = "./sample_data/config/sample_config_seg.ini",
            **kwargs
        )

    def override_config(self):
        override_dict = {
            ('LoaderParams', 'inf_samples_per_vol'): 10,
            ('RunParams', 'Batch_size'): 15
        }
        super(TestSegmentationInferencer, self).override_config(override_dict)


class TestClassificationInferencer(TestInferencer):
    def __init__(self, *args, **kwargs):
        super(TestClassificationInferencer, self).__init__(
            *args,
            sample_config = "./sample_data/config/sample_config_class.ini",
            **kwargs
        )


class TestBinaryClassificationInferencer(TestInferencer):
    def __init__(self, *args, **kwargs):
        super(TestBinaryClassificationInferencer, self).__init__(
            *args,
            sample_config = "./sample_data/config/sample_config_binaryclass.ini",
            **kwargs
        )

class TestrAIdiologistInferencer(TestInferencer):
    def __init__(self, *args, **kwargs):
        super(TestrAIdiologistInferencer, self).__init__(
            *args,
            sample_config = "./sample_data/config/sample_config_rAIdiologist.ini",
            **kwargs
        )

    def test_inference(self):
        super(TestrAIdiologistInferencer, self).test_inference()
        self.assertTrue(len(list(Path(self.temp_output_dir.name).glob("*"))) != 0)