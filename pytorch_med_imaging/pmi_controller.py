# System
import ast
import argparse
import os, gc
import re
import logging
import datetime
import inspect
import configparser
from typing import Optional, Iterable, Union, Any
from pathlib import Path

# Propietary
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import configparser
import numpy as np
from .PMI_data_loader import PMIBatchSamplerFactory, PMIDataFactory
from .PMI_data_loader.pmi_dataloader_base import PMIDataLoaderBase
from .med_img_dataset import PMITensorDataset

# This package
from .networks import *
from .networks.third_party_nets import *
from .networks.specialized import *
from .tb_plotter import TB_plotter
from .solvers import *
from .inferencers import *

from mnts.mnts_logger import MNTSLogger
from tensorboardX import SummaryWriter
import torch.autograd as autograd
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
autograd.set_detect_anomaly(True)

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight, 1)
        m.bias.data.fill_(0.01)

def parse_ini_filelist(filelist, mode):
    assert os.path.isfile(filelist)

    fparser = configparser.ConfigParser()
    fparser.read(filelist)

    # test
    if mode:
        return fparser['FileList'].get('testing').split(',')
    else:
        return fparser['FileList'].get('training').split(',')


class backward_compatibility(object):
    def __init__(self, train, input, lsuffix, loadbyfilelist):
        super(backward_compatibility, self).__init__()
        self.train = train
        self.input = input
        self.lsuffix = lsuffix
        self.loadbyfilelist = loadbyfilelist


class PMIController(object):
    def __init__(self, config: Any, a: argparse.Namespace):
        self.logger = MNTSLogger[self.__class__.__name__]

        # populate attributes
        self._unpack_config(config)
        if not a is None:
            self.override_config(a)
        self.logger.info("Recieve arguments: %s"%dict(({section: dict(self.config[section]) for section in self.config.sections()})))


        # Try to make outputdir first if it exist
        if self.data_output_dir.endswith('.csv'):
            Path(self.data_output_dir).parent.mkdir(exist_ok=True)
        else:
            Path(self.data_output_dir).mkdir(exist_ok=True)
        # simple error check
        self._error_check()

        # Create net
        try:
            if re.search("[\W]+", self.network_network_type.translate(str.maketrans('', '', "(), "))) is not None:
                raise AttributeError(f"You net_nettype specified ({self.network_network_type}) contains illegal characters!")
            self.net = eval(self.network_network_type)
            so = re.search('.+?(?=\()', self.network_network_type)
            if not so is None:
                self.net_name = so.group()
            else:
                self.net_name = "<unknown_network>"
        except:
            raise AttributeError(f"Failed to create network from name: {self.network_network_type}")

        # Prepare data object
        try:
            self.pmi_factory = PMIDataFactory()
            self.pmi_data = self.pmi_factory.produce_object(self.config)
        except Exception as e:
            self.logger.exception("Error creating target object!", logging.FATAL)
            self.logger.error("Original error: {}".format(e))
            return

        self.validation_FLAG=False
        if not self.filters_validation_id_list is None and self.general_plot_tb:
            self.logger.log_print_tqdm("Recieved validation parameters.")
            val_config = configparser.ConfigParser()
            val_config.read_dict(self.config)
            val_config.set('Filters', 're_suffix', self.filters_validation_re_suffix)
            val_config.set('Filters', 'id_list', self.filters_validation_id_list)
            val_config['Data']['input_dir']  = str(self.data_validation_input_dir)
            val_config['Data']['target_dir'] = str(self.data_validation_gt_dir)
            self.pmi_data_val = self.pmi_factory.produce_object(val_config)
            self.validation_FLAG=True

    def create_solver(self, run_type: str) -> SolverBase:
        # check run_type is safe from attacks
        if re.search("[\W]+", run_type) is not None:
            raise ArithmeticError(f"Your run_type ({run_type}) contains illegal characters!")
        solver_class = eval(f'{run_type}Solver')
        self.logger.info("Creating solver: {}".format(solver_class.__name__))
        solver = solver_class(self.net, # created in __init__
                              self._pack_config('SolverParams'),
                              self.general_use_cuda
                              )
        return solver

    def create_inferencer(self, run_type: str) -> InferencerBase:
        # check run_type is safe from attacks
        if re.search("[\W]+", run_type) is not None:
            raise ArithmeticError(f"Your run_type ({run_type}) contains illegal characters!")
        infer_class = eval(f'{run_type}Inferencer')
        self.logger.info("Creating solver: {}".format(infer_class.__name__))
        infer = infer_class(self.runparams_batch_size,
                            self.net,
                            self.checkpoint_cp_load_dir,
                            self.data_output_dir,
                            self.general_use_cuda,
                            self.pmi_data,
                            self.config)
        return infer

    def training(self) -> None:
        #-------------------------------
        # Create training solver and net
        solver = self.create_solver(self.general_run_type)
        loader, loader_val = self.prepare_loaders()

        # Set learning rate scheduler, TODO: move this to solver
        if self.solverparams_decay_on_plateau:
            self.logger.log_print_tqdm("Optimizer decay on plateau.")
            _lr_scheduler_dict = eval(self.solverparams_lr_scheduler_dict)
            if not isinstance(_lr_scheduler_dict, dict):
                self.logger.error("lr_scheduler_dict must eval to a dictionary! Got {} instead.".format(_lr_scheduler_dict))
                return
            self.logger.debug("Got lr_schedular_dict: {}.".format(self.solverparams_lr_scheduler_dict))
            solver.set_lr_decay_to_reduceOnPlateau(3, param_decay, **_lr_scheduler_dict)
        else:
            solver.set_lr_decay_exp(self.solverparams_decay_rate_lr)

        # Push dataloader to solver
        solver.set_dataloader(loader, loader_val)

        # Read tensorboard dir from env, disable plot if it fails
        self.prepare_tensorboard_writter()

        # create plotters
        if self.general_plot_tb:
            solver.set_plotter(self.writer)

        # Load Checkpoint or create new network
        #-----------------------------------------
        # net = solver.get_net()
        solver.get_net().train()
        solver.load_checkpoint(self.checkpoint_cp_load_dir)

        solver.fit(self.checkpoint_cp_save_dir,
                   debug_validation=self.a.debug_validation) # TODO: move checkpoint_save argument to else where
        self.solver = solver

    def prepare_tensorboard_writter(self) -> None:
        r"""Prepare the Tensorboard writer if `general_plot_tb` is set to True. The writer will be automatically
        named based on the timestamp of when the run is initiated. In the future, I might allow the control of the
        directory to write the Tensorboard summary."""
        if self.general_plot_tb:
            try:
                tensorboard_rootdir =  Path(os.environ.get('TENSORBOARD_LOGDIR', '/media/storage/PytorchRuns'))
                if not tensorboard_rootdir.is_dir():
                    self.logger.warning("Cannot read from TENORBOARD_LOGDIR, retreating to default path...")
                    tensorboard_rootdir = Path("/media/storage/PytorchRuns")

                # Strip the parenthesis and comma from the net name to avoid conflicts with system
                net_name = self.network_network_type.translate(str.maketrans('(),','[]-'," "))
                self.logger.info("Creating TB writer, writing to directory: {}".format(tensorboard_rootdir))
                writer = SummaryWriter(tensorboard_rootdir.joinpath(
                    "%s-%s" % (net_name,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))).__str__()
                )
                self.writer = TB_plotter(writer)
            except Exception as e:
                self.logger.warning("Tensorboard writter creation encounters failure, falling back to no writer.")
                self.logger.exception(e)
                self.writer = None
                self.general_plot_tb = False
        else:
            self.writer = None
            self.general_plot_tb = False

    def inference(self):
        self.logger.log_print_tqdm("Starting evaluation...")
        loader, loader_val = self.prepare_loaders()

        #------------------------
        # Create testing inferencer
        if run_type == 'Segmentation':
            infer_class = SegmentationInferencer
        elif run_type == 'Classification':
            infer_class = ClassificationInferencer
        elif run_type == 'BinaryClassification':
            infer_class = BinaryClassificationInferencer
        elif run_type == 'BinaryClassificationRNN':
            infer_class = BinaryClassificationRNNInferencer
        elif run_type == 'Survival':
            infer_class = SurvivalInferencer
        elif run_type == 'rAIdiologist':
            infer_class = rAIdiologistInferencer
        else:
            self.logger.error('Wrong run_type setting!')
            raise NotImplementedError("Not implemented inference type: {}".format(run_type))

        inferencer = infer_class(param_batchsize,
                                 net,
                                 checkpoint_load,
                                 dir_output,
                                 bool_usecuda,
                                 pmi_data,
                                 config)


        # Pass PMI to inferencer if its specified
        if not data_pmi_loader_kwargs is None:
            logger.info("Overriding loader, setting to: {}".format(data_pmi_loader_kwargs))
            loader_factory = PMIBatchSamplerFactory()
            loader = loader_factory.produce_object(inputDataset, config)
            inferencer.set_dataloader(loader)
            logger.info("New loader type: {}".format(loader.__class__.__name__))

        if write_mode == 'GradCAM':
            #TODO: custom grad cam layers
            inferencer.grad_cam_write_out(['att2'])
        else:
            with torch.no_grad():
                if a.inference_all_checkpoints:
                    try:
                        inferencer.write_out_allcps()
                    except AttributeError:
                        logger.warning("Falling back to normal inference.")
                        inferencer.write_out()
                else:
                    inferencer.write_out()

        # Output summary of results if implemented
        if not hasattr(inferencer, 'display_summary'):
            logger.info(f"No summary for the class: {inferencer.__class__.__name__}")
        try:
            inferencer.display_summary()
        except AttributeError as e:
            logger.exception("Error when computing summary.")

    def prepare_loaders(self) -> Iterable[PMIDataLoaderBase]:
        r"""This creates the loader, i.e. torchio iterables for the DataLoader"""
        self.logger.log_print_tqdm("Start training...")
        trainingSubjects = self.pmi_data.load_dataset()
        validationSubjects = self.pmi_data_val.load_dataset() if self.validation_FLAG else (None, None)
        # Prepare dataset
        # numcpu = int(os.environ.get('SLURM_CPUS_ON_NODE', default=torch.multiprocessing.cpu_count()))
        if self.loaderparams_pmi_loader_name is None:
            # num_workers is required to be zero by torchio
            loader = DataLoader(trainingSubjects,
                                batch_size  = self.runparams_batch_size,
                                shuffle     = True,
                                num_workers = 0,
                                drop_last   = True,
                                pin_memory  = False)
            loader_val = DataLoader(validationSubjects,
                                    batch_size  = self.runparams_batch_size,
                                    shuffle     = False,
                                    num_workers = 0,
                                    drop_last   = False,
                                    pin_memory  = False) if self.validation_FLAG else None
        else:
            self.logger.info("Loading custom dataloader.")
            loader_factory = PMIBatchSamplerFactory()
            loader = loader_factory.produce_object(trainingSet, self.config)
            loader_val = loader_factory.produce_object(valSet, self.config,
                                                       force_inference=True) if self.validation_FLAG else None
        return loader, loader_val

    def _error_check(self):
        # Error check
        # -----------------
        # Check directories
        for key in list(self.config['Data']):
            d = Path(self.config['Data'].get(key))
            if not d.is_file() and not d.is_dir() and not d == "":
                if d.endswith('.csv'):
                    continue

                # Only warn instead of terminate for some path if in inference mode
                if self.mode == 1 & (key.find('target') or key.find('gt')):
                    self.logger.warning("Cannot locate %s: %s" % (key, d))
                else:
                    msg = "Cannot locate %s: %s" % (key, d)
                    raise IOError(msg)
        assert Path(self.data_input_dir).is_dir(), "Input data directory not exist!"
        if (self.data_target_dir is None) & self.mode == 0:
            self.logger.warning("No target dir set but specified training mode. Are you sure you are doing "
                                "unsupervised learning?")
            # self.mode = 1 # Eval mode

        available_run_types = (
            'Segmentation',
            'Classification',
            'BinaryClassification',
            'BinaryClassificationRNN',
            'Survival',
            'rAIdiologist',
        )
        if self.general_run_type not in available_run_types:
            msg = f"Wrong run type, got {self.general_run_type} but it must be one of [{'|'.join(available_run_types)}]"
            raise AttributeError(msg)

    def override_config(self, a: Iterable[Any]) -> None:
        r"""This will change and override the ini config. This functino should be called after _unpack_config().
        Note that the --override from argparse is handled here, but the changes to logging will have no effects."""
        # Config override
        # -----------------
        # Updated parameters need to be written back into config
        if a.train:
            self.config['General']['run_mode'] = 'training'
        if a.inference:
            self.config['General']['run_mode'] = 'inference'
        if a.batch_size:
            self.config['RunParams']['batch_size'] = str(a.batch_size)
        if a.epoch:
            self.config['RunParams']['num_of_epochs'] = str(a.epoch)
        if a.lr:
            self.config['RunParams']['learning_rate'] = str(a.lr)
        if a.debug_validation:
            self.config['General']['debug']            = "yes"
            self.config['General']['debug_validation'] = "yes"
        if a.debug:
            self.config['General']['debug'] = "yes"
        if not a.override == '':
            for substring in a.override.split(';'):
                substring = substring.replace(' ', '')
                mo = re.match("\((?P<section>.+),(?P<key>.+)\)\=(?P<value>.+)",substring)
                if mo is None:
                    self.logger.warning("Overriding failed for substring {}".format(substring))
                else:
                    mo_dict = mo.groupdict()
                    _section, _key, _val = [mo_dict[k] for k in ['section', 'key', 'value']]
                    self.logger.info(f"Overrided: ({_section},{_key})={_val}")
                    if not _section in self.config:
                        self.config.add_section(_section)
                    self.config.set(_section, _key, _val)
        self.a = a
        self._unpack_config(self.config)

    def _unpack_config(self, config: Union[str, Path, configparser.ConfigParser]) -> None:
        r"""Unpack the config into attributes. If the argument is not a `ConfigParser` object, this function will try to
        read the config. Noting checking of keys are donw elsewhere in solvers and inferencers

        Args:
            config (str, path or configparser.Configparser):
                If a string or path was supplied, the ini file will be loaded using the config parser. Otherwise, this
                the config is directly unpacked.

        Returns:
            None
        """

        if not isinstance(config, configparser.ConfigParser):
            config_obj = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
            config_obj.read(config)
            config = config_obj

        CASTING_KEYS = {
            ('SolverParams', 'learning_rate')    : float,
            ('SolverParams', 'momentum')         : float,
            ('SolverParams', 'num_of_epochs')    : int,
            ('SolverParams', 'decay_rate_lr')    : float,
            ('SolverParams', 'lr_scheduler_dict'): dict,
            ('SolverParams', 'early_stop')       : dict,
            ('RunParams'   , 'batch_size')       : int,
            ('General'     , 'plot_tb')          : bool,
            ('General'     , 'use_cuda')         : bool,
            ('General'     , 'debug')            : bool,
            ('General'     , 'debug_validation') : bool,
            ('RunParams'   , 'decay_on_plateau') : bool
        }
        DEFAULT_DICT = {
            ('General'     , 'run_mode')            : 'training',
            ('General'     , 'run_type')            : 'segmentation',
            ('Checkpoint'  , 'cp_load_dir')         : "",
            ('Checkpoint'  , 'cp_save_dir')         : "",
            ('Filters'     , 're_suffix')           : "(.*)",
            ('Data'        , 'validation_input_dir'): ('Data'       , 'input_dir'),
            ('Data'        , 'validation_gt_dir')   : ('Data'       , 'target_dir'),
            ('Filters'     , 'validation_re_suffix'): ('Filters'    , 're_suffix'),
            ('RunParams'   , 'initial_weight')      : None,
            ('Network'     , 'initialization')      : None,
            ('Filters'     , 'id_list')             : None,
            ('Filters'     , 'validation_id_list')  : None,
            ('LoaderParams', 'PMI_loader_name')     : None,
            ('LoaderParams', 'PMI_loader_kwargs')   : None,
        }

        # read_dict from
        att_dict = {}
        for sections in config.sections():
            for keys in config[sections]:
                dict_key = (sections, keys)
                att_key = "_".join(dict_key).lower()
                if dict_key in {k: v for k, v in CASTING_KEYS if v == bool}:
                    val = config[sections].getboolean(keys, False)
                else:
                    val = config[sections].get(keys)

                # type case if it is specified
                if dict_key in CASTING_KEYS:
                    if CASTING_KEYS[dict_key] == dict:
                        val = ast.literal_eval(val)
                    else:
                        val = CASTING_KEYS[dict_key](val)

                # if it starts with "{' automatically make this a dictionary
                if isinstance(val, str):
                    try:
                        val = ast.literal_eval(val)
                    except:
                        pass
                att_dict[att_key.lower()] = val

        # read default if they don't exist
        for dict_key in DEFAULT_DICT:
            att_key = "_".join(dict_key).lower()
            if att_key not in att_dict: # load default value if its not specified in the ini file
                # If its a tuple, trace the source key
                if isinstance(DEFAULT_DICT[dict_key], tuple):
                    _src_sec, _src_key = DEFAULT_DICT[dict_key]
                    val = config[_src_sec][_src_key]
                else:
                    val = DEFAULT_DICT[dict_key]
                att_dict[att_key] = val
                config[dict_key[0]][dict_key[1]] = str(val)
            else:
                continue

        self.__dict__.update({'_'.join(k).lower(): v for k, v in DEFAULT_DICT.items()})
        self.__dict__.update(att_dict)
        self.mode = self.general_run_mode in ('testing', 'test')
        self.config = config
        return att_dict

    def _pack_config(self, sections: Union[str, Iterable[str]]) -> dict:
        r"""Pack loaded configs with the same subheader into a dictionary"""
        out_keys = []
        if isinstance(sections, (list, tuple)):
            sections = [s.lower() for s in sections]
        elif isinstance(sections, str):
            sections = [sections.lower()]
        else:
            raise TypeError("Input must be string or iterable of strings.")
        for keys in self.__dict__:
            if re.match("^("+"|".join(sections)+"){1}_\w*", keys) is not None:
                out_keys.append(keys)
        return {k: self.__dict__[k] for k in out_keys}

    def _get_from_config(self, item):
        r"""This override allow getting the access of options defined in the INI config"""
        try:
            self.__dict__[item]
        except KeyError as e:
            if isinstance(item, str):
                self._logger.warning(f"Trying to get un-sepcified attribute: {item}")
                ro = re.search("^(?P<section>[a-zA-Z]+)_(?P<key>.*)", item)
                section = ro.groupdict()['section']
                if ro is None:
                    raise e

                if section in (s.lower() for s in self.config.sections()):
                    section = self.config.sections()[[l.lower() for l in self.config.sections()].index(section)]
                return self.config[section].get(ro.groupdict()['key'], None)

    def run(self):
        r"""Kick start based on the setting from the INI file"""
        if self.mode:
            self.inference()
        else:
            self.training()


def console_entry(raw_args=None):
    parser = argparse.ArgumentParser(description="Training reconstruction from less projections.")
    parser.add_argument("--config", metavar='config', action='store', required=True,
                        help="Config .ini file.", type=str)
    parser.add_argument("-t", "--train", dest='train', action='store_true', default=False,
                        help="Set this to force training mode. (Implementing)")
    parser.add_argument("-i", "--inference", dest="inference", action='store_true', default=False,
                        help="Set this to force inference mode. If used with -t option, will still go into inference. (Implementing")
    parser.add_argument("-b", "--batch-size", dest='batch_size', type=int, default=None,
                        help="Set this to override batch-size setting in loaded config.")
    parser.add_argument("-e", "--epoch", dest="epoch", type=int, default=None,
                        help="Set this to override number of epoch when loading config.")
    parser.add_argument("-l", "--lr", dest='lr', type=float, default=None,
                        help="Set this to override learning rate.")
    parser.add_argument("--all-checkpoints", dest='inference_all_checkpoints', action='store_true',
                        help="Set this to inference all checkpoints.")
    parser.add_argument("--log-level", dest='log_level', type=str, choices=('debug', 'info', 'warning','error'),
                        default='info', help="Set log-level of the logger.")
    parser.add_argument("--keep-log", action='store_true',
                        help="If specified, save the log file to the `log_dir` specified in the config.")
    parser.add_argument('--debug', dest='debug', action='store_true', default=None,
                        help="Set this to initiate the config with debug setting.")
    parser.add_argument('--debug-validation', action='store_true',
                        help="Set this to true to run validation direction. This also sets --debug to true.")
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help="Print message to stdout.")
    parser.add_argument('--override', dest='override', action='store', type=str, default='',
                        help="Use syntax '(section1,key1)=value1;(section2,key2)=value' to override any"
                             "settings specified in the config file. Note that no space is allowed.")

    a = parser.parse_args(raw_args)

    assert os.path.isfile(a.config), f"Cannot find config file {a.config}! Curdir: {os.listdir('.')}"

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(a.config)

    # Override config settings, move this into override?
    pre_log_message = []

    # Parameters check
    log_dir = config['General'].get('log_dir', './Backup/Log/')
    keep_log = config['General'].getboolean('keep_log', False)
    if not Path(log_dir).parent.is_dir():
        Path(log_dir).parent.mkdir(parents=True, exist_ok=True)
        pass
    if os.path.isdir(log_dir):
        print(f"Log file not designated, creating under {log_dir}")
        log_dir = os.path.join(log_dir, "%s_%s.log"%(config['General'].get('run_mode', 'training'),
                                                     datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    print(f"Log designated to {log_dir}")
    print(f"Fullpath: {os.path.abspath(log_dir)}")
    with MNTSLogger(log_dir, logger_name='pmi-main', verbose=a.verbose, keep_file=keep_log,
                    log_level='debug' if a.debug else 'debug') as logger:
        logger.info("Global logger: {}".format(logger))

        for msg in pre_log_message:
            logger.info(msg)

        logger.info(">" * 40 + " Start Main " + "<" * 40)
        try:
            main = PMIController(config, a)
            main.run()
        except Exception as e:
            logger.error("Uncaught exception!")
            logger.exception(e)
            raise BrokenPipeError("Unexpected error in main().")
        logger.info("=" * 40 + " Done " + "="* 40)

if __name__ == '__main__':
    console_entry()