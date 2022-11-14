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
    r"""This controller is the backbone of the package. To initiate training or inference, you will need to
    first create an INI config file, using which an instance of PMIController can be created. Then simply call
    either :func:`training()` or :func:`inference()` to run the code.

    The INI file is first parsed using :func:`_unpack_config`. The definitions of data type and default values
    are hard-coded there.

    Attributes:
        solver (SolverBase):
            If the running mode is set to "training", the PMIController will create a :class:`SolverBase` instance
            duing :func:`training()` according to the option `('General', 'run_type')`
        inferencer (InferencerBase):
            Similar to solver, but it is created during :func:`inference()` instead.
    """
    def __init__(self, config: Any, a: argparse.Namespace):
        self._logger = MNTSLogger[self.__class__.__name__]

        # populate attributes
        self._unpack_config(config)
        if not a is None:
            self.override_config(a)
        self._logger.info("Recieve arguments: %s" % dict(({section: dict(self.config[section]) for section in self.config.sections()})))


        # Try to make outputdir first if it exist
        if self.data_output_dir.endswith('.csv'):
            Path(self.data_output_dir).parent.mkdir(exist_ok=True)
        else:
            Path(self.data_output_dir).mkdir(exist_ok=True, parents=True)
        # simple error check
        self._error_check()

        # Create net
        try:
            if re.search("[\W]+", self.network_network_type.translate(str.maketrans('', '', "(), =\."))) is not None:
                raise AttributeError(f"You net_nettype specified ({self.network_network_type}) contains illegal characters!")
            self.net = eval(self.network_network_type)
            so = re.search('.+?(?=\()', self.network_network_type)
            if not so is None:
                self.net_name = so.group()
            else:
                self.net_name = "<unknown_network>"
        except:
            raise AttributeError(f"Failed to create network from name: {self.network_network_type}")

        # Try to load pretrain checkpoint if specified
        if self.network_pretrain_cp != "":
            self._logger.info("Loading pretrained network.")
            try:
                self.net.load_state_dict(torch.load(self.network_pretrain_cp))
            except Exception as e:
                self._logger.exception(e)
                self._logger.warning("Pretrain network specified but cannot be loaded.")

        # Prepare data object
        try:
            self.pmi_factory = PMIDataFactory()
            self.pmi_data = self.pmi_factory.produce_object(self.config)
        except Exception as e:
            self._logger.exception("Error creating target object!", logging.FATAL)
            self._logger.error("Original error: {}".format(e))
            return

        # If validation flags were set, create data loaders required for validation
        self.validation_FLAG=False
        if not self.filters_validation_id_list in (None , "") and self.general_plot_tb:
            self._logger.log_print_tqdm("Recieved validation parameters.")
            val_config = configparser.ConfigParser()
            val_config.read_dict(self.config)
            val_config.set('Filters', 're_suffix', self.filters_validation_re_suffix)
            if a is not None: # --validate-on-test-set option
                if a.validate_on_test_set:
                    val_config.set('Filters', 'id_list', self.filters_id_list)
                    # This will trick pmi_data to load from .ini the testing row
                    val_config.set('General', 'run_mode', 'inference')
                else:
                    val_config.set('Filters', 'id_list', self.filters_validation_id_list)
            val_config['Data']['input_dir']  = str(self.data_validation_input_dir)
            val_config['Data']['target_dir'] = str(self.data_validation_gt_dir)
            self.pmi_data_val = self.pmi_factory.produce_object(val_config)
            self.validation_FLAG=True

    def create_solver(self, run_type: str) -> SolverBase:
        r"""This is a pseudo factory that produces a solver instance based on the name specified.

        Args:
            run_type (str):
                The name of the solver, no need to include the term "Solver".
        Returns:
            SolverBase
        """
        # check run_type is safe from attacks
        run_type = run_type.replace("Solver", "") # more flexible
        if re.search("[\W]+", run_type) is not None:
            raise ArithmeticError(f"Your run_type ({run_type}) contains illegal characters!")
        solver_class = eval(f'{run_type}Solver')
        self._logger.info("Creating solver: {}".format(solver_class.__name__))
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
        self._logger.info("Creating solver: {}".format(infer_class.__name__))
        infer = infer_class(self.net,
                            self.data_output_dir,
                            self._pack_config('SolverParams'),
                            self.general_use_cuda,
                            self.pmi_data,
                            debug=self.general_debug)
        return infer

    def training(self) -> None:
        #-------------------------------
        # Create training solver and net
        self._logger.info("Start training...")
        self._logger.info(f"Creating solver: {self.general_run_type}")
        solver = self.create_solver(self.general_run_type)
        loader, loader_val = self.prepare_loaders()
        # Push dataloader to solver
        solver.set_dataloader(loader, loader_val)

        # Set learning rate scheduler, TODO: move this to solver
        if self.solverparams_decay_on_plateau:
            self._logger.warning("Decay on plateau option is deprecated.")

        if self.solverparams_lr_scheduler is not None:
            self.create_lr_scheduler(solver)

        else:
            self._logger.info("LR scheduler not specified, using default exponential.")
            solver.set_lr_decay_exp(self.solverparams_decay_rate_lr)

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

        try:
            debug_validation = self.a.debug_validation
        except AttributeError:
            debug_validation = False
        solver.fit(self.checkpoint_cp_save_dir,
                   debug_validation=debug_validation) # TODO: move checkpoint_save argument to else where
        self.solver = solver

    def create_lr_scheduler(self, solver):
        self._logger.info(f"Creating lr_scheduler: {self.solverparams_lr_scheduler}.")
        if isinstance(self.solverparams_lr_scheduler_args, (float, int, str)):
            self.solverparams_lr_scheduler_args = [self.solverparams_lr_scheduler_args]
        solver.set_lr_scheduler(self.solverparams_lr_scheduler,
                                *self.solverparams_lr_scheduler_args,
                                **self.solverparams_lr_scheduler_kwargs)

    def prepare_tensorboard_writter(self) -> None:
        r"""Prepare the Tensorboard writer if `general_plot_tb` is set to True. The writer will be automatically
        named based on the timestamp of when the run is initiated. In the future, I might allow the control of the
        directory to write the Tensorboard summary."""
        if self.general_plot_tb:
            try:
                tensorboard_rootdir =  Path(os.environ.get('TENSORBOARD_LOGDIR', '/media/storage/PytorchRuns'))
                if not tensorboard_rootdir.is_dir():
                    self._logger.warning("Cannot read from TENORBOARD_LOGDIR, retreating to default path...")
                    tensorboard_rootdir = Path("/media/storage/PytorchRuns")

                # Strip the parenthesis and comma from the net name to avoid conflicts with system
                net_name = self.network_network_type.translate(str.maketrans('(),','[]-'," "))
                self._logger.info("Creating TB writer, writing to directory: {}".format(tensorboard_rootdir))
                writer = SummaryWriter(tensorboard_rootdir.joinpath(
                    "%s-%s" % (net_name,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))).__str__()
                )
                self.writer = TB_plotter(writer)
            except Exception as e:
                self._logger.warning("Tensorboard writter creation encounters failure, falling back to no writer.")
                self._logger.exception(e)
                self.writer = None
                self.general_plot_tb = False
        else:
            self.writer = None
            self.general_plot_tb = False

    def inference(self):
        self._logger.log_print_tqdm("Starting evaluation...")
        loader = self.prepare_loaders(inference_mode=True)
        inferencer = self.create_inferencer(self.general_run_type)
        inferencer.set_dataloader(loader)
        inferencer.load_checkpoint(self.checkpoint_cp_load_dir)

        with torch.no_grad():
            # if a.inference_all_checkpoints:
            #     try:
            #         inferencer.write_out_allcps()
            #     except AttributeError:
            #         logger.warning("Falling back to normal inference.")
            #         inferencer.write_out()
            # else:
            inferencer.write_out()

        # Output summary of results if implemented
        if not hasattr(inferencer, 'display_summary'):
            self._logger.info(f"No summary for the class: {inferencer.__class__.__name__}")
        try:
            inferencer.display_summary()
        except AttributeError as e:
            self._logger.exception("Error when computing summary.")

    def prepare_loaders(self, inference_mode = False) -> Iterable[PMIDataLoaderBase]:
        r"""This creates the loader, i.e. torchio iterables for the DataLoader"""
        if inference_mode: # switch to inference loading mode
            _tmp = self.pmi_data._run_mode
            self.pmi_data._run_mode = 'inference'

        if inference_mode and self.a.inference_on_validation_set:
            self._logger.info("Inferencing on validation set")
            subjects = self.pmi_data_val.load_dataset(exclude_augment=True) if self.validation_FLAG else (None, None)
        elif inference_mode and self.a.validate_on_test_set:
            subjects = self.pmi_data.load_dataset(exclude_augment=True)
        else:
            subjects = self.pmi_data.load_dataset()

        # Prepare dataset
        # numcpu = int(os.environ.get('SLURM_CPUS_ON_NODE', default=torch.multiprocessing.cpu_count()))
        if self.loaderparams_pmi_loader_name is None:
            # num_workers is required to be zero by torchio
            loader = DataLoader(subjects,
                                batch_size  = self.runparams_batch_size,
                                shuffle     = not inference_mode,
                                num_workers = 0,
                                drop_last   = not inference_mode,
                                pin_memory  = False)
            if not inference_mode:
                # TODO: Need to handle when there are no validation data
                self.pmi_data_val._run_mode = 'inference_mode'
                validationSubjects = self.pmi_data_val.load_dataset() if self.validation_FLAG else (None, None)
                loader_val = DataLoader(validationSubjects,
                                        batch_size  = self.runparams_batch_size,
                                        shuffle     = False,
                                        num_workers = 0,
                                        drop_last   = False,
                                        pin_memory  = False) if self.validation_FLAG else None
        else:
            self._logger.info("Loading custom dataloader.")
            raise NotImplementedError
            # loader_factory = PMIBatchSamplerFactory()
            # loader = loader_factory.produce_object(trainingSet, self.config)
            # loader_val = loader_factory.produce_object(valSet, self.config) if self.validation_FLAG else None

        if inference_mode:
            self.pmi_data._run_mode = _tmp
            return loader
        else:
            return loader, loader_val

    def _error_check(self):
        # Error check
        # -----------------
        # Check directories
        for key in list(self.config['Data']):
            d = Path(self.config['Data'].get(key))
            if not d.is_file() and not d.is_dir() and not d == "":
                if d.suffix == '.csv':
                    continue

                # Only warn instead of terminate for some path if in inference mode
                if self.mode == 1 & (key.find('target') or key.find('gt')):
                    self._logger.warning("Cannot locate %s: %s" % (key, d))
                else:
                    msg = "Cannot locate %s: %s" % (key, d)
                    raise IOError(msg)
        assert Path(self.data_input_dir).is_dir(), f"Input data directory {self.data_input_dir} not exist!"
        if (self.data_target_dir is None) & self.mode == 0:
            self._logger.warning("No target dir set but specified training mode. Are you sure you are doing "
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
            self.config['SolverParams']['num_of_epochs'] = str(a.epoch)
        if a.lr:
            self.config['SolverParams']['learning_rate'] = str(a.lr)
        if a.debug_validation:
            self.config['General']['debug']            = "yes"
            self.config['General']['debug_validation'] = "yes"
        if a.debug:
            self.config['General']['debug'] = "yes"
        if a.fold_code not in (None, ""):
            self.config['General']['fold_code'] = a.fold_code
        if a.network not in (None, ""):
            self.config['Network']['network_type'] = a.network
        if a.inference_on_training_set:
            self.config['General']['force_train_data'] = 'yes'
        if a.inference_on_validation_set:
            self.config['General']['force_validation_data'] = 'yes'
        if not a.override == '':
            for substring in a.override.split(';'):
                substring = substring.replace(' ', '')
                mo = re.match("\((?P<section>.+),(?P<key>.+)\)\=(?P<value>.+)",substring)
                if mo is None:
                    self._logger.warning("Overriding failed for substring {}".format(substring))
                else:
                    mo_dict = mo.groupdict()
                    _section, _key, _val = [mo_dict[k] for k in ['section', 'key', 'value']]
                    # ' and " charaters are not allowed
                    _section = re.sub("('|\")", "", _section)
                    _key = re.sub("('|\")", "", _key)
                    self._logger.info(f"Overrided: ({_section},{_key})={_val}")
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
            ('SolverParams', 'learning_rate')      : float,
            ('SolverParams', 'momentum')           : float,
            ('SolverParams', 'num_of_epochs')      : int,
            ('SolverParams', 'accumulate_grad')    : int,
            ('SolverParams', 'decay_rate_lr')      : float,
            ('SolverParams', 'lr_scheduler_dict')  : dict,
            ('SolverParams', 'lr_scheduler_kwargs'): dict,
            ('SolverParams', 'early_stop')         : dict,
            ('RunParams'   , 'batch_size')         : int,
            ('SolverParams', 'decay_on_plateau')   : bool,  # Deprecated
            ('General'     , 'plot_tb')            : bool,
            ('General'     , 'use_cuda')           : bool,
            ('General'     , 'debug')              : bool,
            ('General'     , 'debug_validation')   : bool,
        }
        DEFAULT_DICT = {
            ('General'     , 'run_mode')            : 'training',
            ('General'     , 'run_type')            : 'segmentation',
            ('Checkpoint'  , 'cp_load_dir')         : "",
            ('Checkpoint'  , 'cp_save_dir')         : "",
            ('Filters'     , 're_suffix')           : "(.*)",
            ('Data'        , 'input_dir')           : "",
            ('Data'        , 'target_dir')          : None,
            ('Data'        , 'validation_input_dir'): ('Data'   , 'input_dir'),
            ('Data'        , 'validation_gt_dir')   : ('Data'   , 'target_dir'),
            ('Filters'     , 'validation_re_suffix'): ('Filters', 're_suffix'),
            ('RunParams'   , 'initial_weight')      : None,
            ('Network'     , 'initialization')      : None,
            ('Network'     , 'pretrain_cp')         : "",
            ('Filters'     , 'id_list')             : "",
            ('Filters'     , 'validation_id_list')  : "",
            ('LoaderParams', 'PMI_loader_name')     : None,
            ('LoaderParams', 'PMI_loader_kwargs')   : None,
            ('LoaderParams', 'PMI_datatype_name')   : None,
            ('SolverParams', 'lr_scheduler')        : None,
            ('SolverParams', 'early_stop')          : None,
            ('SolverParams', 'decay_on_plateau')    : False,
            ('General'     , 'plot_tb')             : False,
            ('General'     , 'use_cuda')            : False,
            ('General'     , 'debug')               : False,
            ('General'     , 'debug_validation')    : False,
            ('SolverParams', 'decay_rate_lr')       : 0.99,
            ('SolverParams', 'lr_scheduler_args')   : ('SolverParams', 'decay_rate_lr'),
            ('SolverParams', 'accumulate_grad')     : 0,
        }

        # read_dict from
        att_dict = {}
        for sections in config.sections():
            for keys in config[sections]:
                dict_key = (sections, keys)
                att_key = self._make_dict_key(sections, keys)
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

                # try to eval any string
                if isinstance(val, str):
                    try:
                        val = ast.literal_eval(val)
                    except Exception as e:
                        pass
                att_dict[att_key] = val

        # read default if they don't exist
        for dict_key in DEFAULT_DICT:
            att_key = self._make_dict_key(*dict_key)
            if att_key not in att_dict: # load default value if its not specified in the ini file
                # If its a tuple, trace the source key
                if isinstance(DEFAULT_DICT[dict_key], tuple):
                    _src_sec, _src_key = DEFAULT_DICT[dict_key]
                    try:
                        val = config[_src_sec][_src_key]
                    except:
                        # val = None
                        continue
                else:
                    val = DEFAULT_DICT[dict_key]
                att_dict[att_key] = val
                config[dict_key[0]][dict_key[1]] = str(val)
            else:
                continue

        self.__dict__.update({self._make_dict_key(*k): v for k, v in DEFAULT_DICT.items()})
        self.__dict__.update(att_dict)
        self.mode = self.general_run_mode in ('testing', 'test', 'inference')
        self.config = config
        return att_dict

    @staticmethod
    def _make_dict_key(sections: str, key: str):
        return sections.lower() + "_" + key.lower()

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
        raise NotImplementedError
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


