import os
import re

from ..solvers import SolverBaseCFG, SolverDDPWrapper
from ..pmi_base_cfg import PMIBaseCFG
from ..pmi_data_loader import PMIDataLoaderBase, PMIDataLoaderBaseCFG, PMIDistributedDataWrapper
from ..integration import *
from pathlib import Path
from typing import Union, Optional, Any

import yaml
from mnts.mnts_logger import MNTSLogger

import torch
import torch.distributed as dist

__all__ = ['PMIController', 'PMIControllerCFG']


PathLike = Union[str, Path]

r"""
A simple API is used to override the template setting through a json

"""
class PMIControllerCFG(PMIBaseCFG):
    r"""The master configuration template that controls the whole pipeline.

    Class Attributes:
        [Required] data_loader_cfg (PMIDataLoaderBaseCFG):
            The cfg instance to create the data loader for loading training.
        [Required] solver_cfg (SolverBaseCFG):
            The cfg instance to create the solver
        [Required] data_loader_cls (type):
            The class of the data_loader. This is used to create the data loader instances with ``data_loader_cfg`` and
            ``data_loader_val_cfg`` during DDP.
        [Required] solver_cls (type):
            The class of the solver. This is used to create the solver instance with ``solver_cfg``.
        [Required] output_dir (str or Path):
            Directory to deposit the inference output. Used by :class:`.InferencerBase` and its children.
        data_loader_val_cfg (PMIDataLoaderBaseCFG, Optional):
            The cfg instance to create the validation instance. If this is ``None`` while ``id_list_val`` is not, the
            cfg ``data_loader_cfg`` will be used for loading validation data. Note that this is not safe and not always
            correct becuase sometime validation require different settings as training
        data_loader_val_cls (type, Optional):
            The data class for validation data loader. If this is not specified and ``data_loader_val_cfg`` is not
            ``None``. This will copy the setting from ``data_loader_cls``.
        fold_code (str):
            This is a tag used to identify the current train-test split. If this is not ``None``, this class will
            replace the string ``'{fold_code}'`` in attributes of the solver cfg and data loader cfg. For target
            attributes, see :func:`PMIController._pre_process_flags`.
        run_mode (str):
            Specify 'training' or 'testing'.
        id_list (PathLike):
            Specify the path to the ID list '.ini' file that specify the training and testing IDs. See
            :func:`~pytorch_med_imaging.SolverBase.parse_ini_filelist`. If this is not specified, ``None`` will be
            passed to the data loaders, typically all files will be loaded from the targeted ``input_dir``.
        id_list_val (PathLike, Optional):
            Specify the path to the ID list '.txt' file that specify the validation data. Note that this does
            accept '.ini' file.
        debug_mode (bool, Optional):
            This option will be passed to solver and data loader CFG. Default to ``False``.
        debug_validation (bool, Optional):
            Default to ``False``.
        validate_on_test_set (bool, Optional):
            Default to ``False``.
        validate_on_training_set (bool, Optional):
            This option is used for debugging. Default to ``False``.
        inference_on_training_set (bool, Optional):
            Default to ``False``.
        inference_on_validation_set (bool, Optional):
            Default to ``False``.
        inference_all_checkpoints (bool, Optional):
            Default to ``False``.
        log_dir (str, Optional):
            Default directory for outputting the log file. Default to ``'./Backup/Log'``.
        keep_log (bool, Optional):
            Default to ``True``.
        verbose (bool, Optional):
            Default to ``True``.
        matmul_precision (str, Optional):
            Option to control the matrix multiplication precision. Only useful when torch version >= 2.0.0. Default to
            'medium'. Must be one of ('medium', 'high', 'highest').
        plotting (Optional[bool]):
            Indicates whether plotting is enabled. Defaults to `False`.
        plotter (Optional[Any]):
            Specifies the plotter instance if plotting is enabled. Defaults to `False`.
        plotter_type (Optional[str]):
            Defines the type of plotter to be used. Defaults to `None`.

    .. note::
        * Don't confuse the `id_list` in this CFG with that in :class:`SolverBase`, the later is more flexible and can
          accept various specification formats.
        * The solver is used for both training and inference. The attribute :attr:`run_mode` determine which mode it
          will run in.
        * If you need different train and inference dataloader, specify `_data_loader_cfg` and `_data_loader_inf_cfg`,
          the code will automatically recognize these two variable and use thme instead of the one in the solver_cfg.

    .. tips::
        If you would like to use a different data loader CFG in training and inference mode, define the private tag
        `_data_loader_cfg` and `_data_loader_inf_cfg` instead of `data_loader_cfg`. This would the property to return
        a data_loader_cfg based on the runtime :attr:`run_mode`.

    """
    fold_code                  : str            = None
    run_mode                   : str            = 'training'
    id_list                    : PathLike       = None
    id_list_val                : PathLike       = None
    output_dir                 : PathLike       = None
    debug_mode                 : Optional[bool] = False
    debug_validation           : Optional[bool] = False
    validate_on_testing_set    : Optional[bool] = False # this changes the how the validation data loader IDs in training mode
    validate_on_training_set   : Optional[bool] = False # this is generally for debugging
    inference_on_training_set  : Optional[bool] = False # this changes the data loader IDs in the inference mode
    inference_on_validation_set: Optional[bool] = False
    inference_all_checkpoints  : Optional[bool] = False
    matmul_precision           : Optional[str]  = 'medium' # Only effect when torch version >= 2.0.0


    # log related
    log_dir : Optional[str]  = './Backup/Log/'
    keep_log: Optional[bool] = True
    verbose : Optional[bool] = True

    # Configurations
    _data_loader_cfg    : PMIDataLoaderBaseCFG           = None # Use this for different train and inference loader
    _data_loader_inf_cfg: Optional[PMIDataLoaderBaseCFG] = None # Use this for different train and inference loader
    data_loader_val_cfg : Optional[PMIDataLoaderBaseCFG] = None
    solver_cfg         : SolverBaseCFG = None
    data_loader_cls    : type          = None
    _data_loader_val_cls: type          = None
    solver_cls         : type          = None
    inferencer_cls     : type          = None

    # Plotting related
    plotting         : Optional[bool] = False
    plotter          : Optional[Any]  = None
    plotter_type     : Optional[str]  = None
    plotter_init_meta: Optional[dict] = {}
    neptune_id       : Optional[str]  = None

    @property
    def data_loader_cfg(self):
        r"""This property allows you to have a different solver CFG in training and inference mode.
        """
        if self.run_mode == 'training':
            return self._data_loader_cfg
        else:
            return self._data_loader_inf_cfg or self._data_loader_cfg

    @data_loader_cfg.setter
    def data_loader_cfg(self, x):
        r"""This will make sure deep copy works."""
        if self.solver_cfg.run_mode == 'training':
            self._data_loader_cfg = x
        else:
            self._data_loader_inf_cfg = x

    @property
    def data_loader_inf_cfg(self):
        r"""For completeness."""
        return self._data_loader_inf_cfg

    @data_loader_inf_cfg.setter
    def data_loader_inf_cfg(self, x):
        r"""For completeness."""
        self._data_loader_inf_cfg = x

    @property
    def data_loader_val_cls(self):
        r"""If validation cls was not set, use the data_loader_cls instead."""
        return self._data_loader_val_cls or self.data_loader_cls

    @data_loader_val_cls.setter
    def data_loader_val_cls(self, x):
        r"""For completeness"""
        self._data_loader_val_cls = x


class PMIController(object):
    r"""The controller to initiate training or inference. Based on the input cfg, this class will create a solver or
    an inferencer, and also the dataloaders.
    """
    def __init__(self, cfg):
        # if global logger is already created, its configurations are not controlled by this controller
        if isinstance(MNTSLogger.global_logger, MNTSLogger):
            self._logger = MNTSLogger[self.__class__.__name__]

        # Load configs
        self._load_config(cfg)

        # otherwise, create the logger based on controller configs
        if not isinstance(MNTSLogger.global_logger, MNTSLogger):
            log_dir = Path(self.log_dir)
            if isinstance(self.solver_cfg.net, torch.nn.Module):
                log_fname = self.solver_cfg.net._get_name()
            elif isinstance(self.solver_cfg.net, str):
                log_fname = self.solver_cfg.net.translate(str.maketrans('(),','[]-'," "))
            else:
                log_fname = 'default'
            if not log_dir.suffix == '.log':
                log_dir = log_dir.joinpath(log_fname)
            existing_logs_indices = [int(re.search(f"{log_fname}-(?P<num_log>\d+)\.log", l.name))
                                     for l in log_dir.parent.glob(f'{log_fname}-*.log')]
            if len(existing_logs_indices) > 0:
                new_log_index = max(existing_log_indices) + 1
                log_dir = log_dir.with_name(f'{log_fname}-{new_log_index}.log')
            else:
                log_dir = log_dir.with_suffix('.log')

            # check if log dir parent is active
            if not log_dir.parent.is_dir():
                if log_dir.parent.is_file():
                    msg = "The target directory specified by log_dir is already created and is a file. "
                    raise FileExistsError(msg)
                log_dir.parent.mkdir(parents=True, exist_ok=True)

            self._logger = MNTSLogger(log_dir, logger_name='pmi_controller', keep_file=self.keep_log,
                                      verbose=self.verbose, log_level='debug')

        self._required_attributes_train = [
            'cp_save_dir'
        ]

        self._required_attributes_inference = [
            'cp_load_dir'
        ]

        # Finally create plotter
        if self.plotting:
            self.create_plotter()
            # if this is a neptune run, add the ID to the config
            if self.plotter_type == 'neptune':
                current_np_id = self._plotter.np_run['sys/id'].fetch()
                with open('./flags.yaml', 'r') as f:
                    flags = yaml.safe_load(f)
                flags['controller_cfg']['neptune_id'] = current_np_id
                with open('./flags.yaml', 'w') as f:
                    yaml.safe_dump(flags, f)
        else:
            self._plotter = None


    def _load_config(self, config_file = None):
        r"""Function to load the configurations. If ``config_file`` is ``None``, load the default class
        :class:`SolverBaseCFG` that is stored as a class attribute :attr:`cls_cfg`.

        Args:
            config_file (PMIControllerCFG):
                The configuration instance.

        """
        # Loading basic inputs
        if not config_file is None:
            cls = config_file
            cls_dict = { attr: getattr(cls, attr) for attr in dir(cls)}
            self.__dict__.update(cls_dict)
            self.__class__.cls_cfg = cls
            self.cfg = config_file
        else:
            if hasattr(self, '_logger'):
                self._logger.warning("_load_config called without arguments.")

        # This is the root that dictates the behavior of the controller.
        if re.match('(?=.*train.*)', self.run_mode) is not None:
            self.run_mode = True
        else:
            self.run_mode = False

    def check_flags_sanity(self):
        _check = [
            (self.solver_cls, type),
            (self.inferencer_cls, type),
        ]
        for k, v in _check:
            if k is None:
                continue
            if not isinstance(k, v):
                msg = f"Expect attribute {k.__name__} to be {v}, got {type(k)} instead."
                raise TypeError

    def override_cfg(self, override_file: PathLike) -> None:
        r"""This method is the key method to control the pipeline execution with the AI experimental manager `guild`.

        Args:
            override_file (str or Path):
                Path to a yaml file to override the cfgs.

        Example:

            example_override.yaml

            .. code-block:: yaml

                controller_cfg:
                    fold_code: 'B00'
                    id_list: './sample_id_list/{fold_code}.ini'
                    id_list_val: './sample_id_list/validation.txt'

                solver_cfg:
                    batch_size: 4
                    init_lr: 1E-5
                    num_of_epoch: 200

            This yaml will override the corresponding attributes of the ``controller`` and the ``solver`` before these
            instances were initialized. The flags can be specified when calling ``guild run`` like this

            .. code-block:: bash
                guild run solver_cfg.batch_size=10 controller_cfg.fold_code=['B00','B01','B02']

        .. important::
            * There are no checking implemented for this override function and the controller will not work if you
              override important attributes such as 'solver' and 'data_loader'. Don't override any attributes that is
              suppose to be an object/instance.

        """
        override_file = Path(override_file)
        if not override_file.is_file():
            msg = f"Expect input to be a directory to a yaml file, got {override_file} but it doesn't point to a file."
            raise FileNotFoundError(msg)

        if override_file.suffix in ('.yml', '.yaml'):
            with override_file.open('r') as f:
                override_dict = yaml.safe_load(f)
                self.guild_dict = override_dict
                self.guild_yaml_path = override_file
        else:
            msg = f"Suffix of the override file should be align with guildai's definitions. Currently only '.ini', " \
                  f"'.yaml' and '.json' were implemented. Got {override_file.name} instead."
            raise ArithmeticError(msg)

        # Global flags
        global_flags = override_dict.get('global_cfg', None)
        if not global_flags is None:
            for  k, v in global_flags.items():
                os.environ[k] = str(v)

        # Override template setting with flags
        controller_override = override_dict.get('controller_cfg', None)
        solver_override = override_dict.get('solver_cfg', None)
        data_loader_override = override_dict.get('data_loader_cfg', None)
        data_loader_val_override = override_dict.get('data_loader_val_cfg', None)
        if not controller_override is None:
            self._override_subcfg(controller_override, self.cfg)
            self._load_config(self.cfg)
        if not solver_override is None:
            self._override_subcfg(solver_override, self.solver_cfg)
        if not data_loader_override is None:
            self._override_subcfg(data_loader_override, self.data_loader_cfg)
        if not data_loader_val_override is None and not self.data_loader_val_cfg is None:
            self._override_subcfg(data_loader_val_override, self.data_loader_val_cfg)


    def _pre_process_flags(self) -> None:
        r"""The flags defined in :class:`PMIControllerCFG` might need to further change the CFGs of the solver and the
        data loader. This method implements this. Essentially, the first part of this method replaces the key tag
        ``'{fold_code}'`` in several attributes with the value stored in :attr:`self.fold_code`. This allows users to
        define K-fold data split and train various folds without having to copy the configuration in each fold. Second,
        this method deals with the special options

        Special options:

        * ``validate_on_testing_set``
        * ``validate_on_training_set``
        * ``inference_on_training_set``
        * ``inference_on_validation_set``

        """
        # Fold code replace filelist and checkpoints
        if not self.fold_code is None:
            # rebuild data loader id lists
            replace_target = [
                (self, 'log_dir'),
                (self, 'cp_load_dir'),
                (self, 'cp_save_dir'),
                (self, 'output_dir'),
                (self, 'id_list'),
                (self.solver_cfg, 'rAI_pretrained_swran')
            ]
            for inst, attr in replace_target:
                _old = getattr(inst, attr, None)
                if _old is None:
                    continue
                _new = _old.replace('{fold_code}', self.fold_code)
                self._logger.debug(f"Replace {_old} with {_new}")
                setattr(inst, attr, _new)

        # if in debug_mode
        if self.debug_mode:
            self._logger.info("Running in debug mode.")
            try:
                self.data_loader_cfg.debug_mode = True
                self.solver_cfg.debug_mode = True
                self.data_loader_val_cfg.debug_mode = True
            except:
                pass

        # if `id_list_val` is defined but `data_loader_val` isn't, try to create it from the `data_loader`
        if getattr(self, 'id_list_val', None) is not None:
            if getattr(self, 'data_loader_val_cfg', None) is None:
                from copy import copy
                self.data_loader_val_cfg = copy(self.data_loader_cfg)
                self.data_loader_val_cls = self.data_loader_cls

        # Read the id lists, note that ``controller.id_list`` and ``data_loader.id_list`` are different in nature
        try:
            if not self.id_list is None:
                testing_ids = PMIDataLoaderBase.parse_ini_filelist(self.id_list, 'testing')
                training_ids = PMIDataLoaderBase.parse_ini_filelist(self.id_list, 'training')
                testing_ids.sort()
                training_ids.sort()
            else:
                # if no IDs provide, perform training/inference on all data within the folder by setting id_list
                # of the loaders to ``None``
                testing_ids = training_ids = None
        except Exception as e:
            if self._logger.log_level == 10:
                self._logger.debug("Encounter error when reading ID filelists")
                self._logger.exception(e)
            msg = f"File IDs must be specified with .ini file with section 'FileList' and attributes 'testing' " \
                  f"and 'training'. Got {self.id_list} but target does not fit the required format."
            raise FileNotFoundError(msg)
        # validation id_lists
        try:
            if not self.id_list_val is None:
                # if a txt file is provided
                if isinstance(self.id_list_val, (str, Path)):
                    self._logger.info(f"Reading validation set ID from: {self.id_list_val}")
                    with open(self.id_list_val, 'r') as _val_txt:
                        validation_ids = [r.rstrip() for r in _val_txt.readlines()]
                    validation_ids.sort()
                else:
                    # if a list of str is provided
                    if not all(isinstance(r, str) for r in self.id_list_val):
                        self._logger.error("If a list is provided for `id_list_val`, all its element must be string.")
                    validation_ids = self.id_list_val
            else:
                self._logger.info("No ID for validation specified.")
                validation_ids = None
        except Exception as e:
            if self._logger.log_level == 10:
                self._logger.debug("Encounter error when reading ID filelists")
                self._logger.exception(e)
            msg = f"Validation ID must be specified with a plain text file with each roll as a unique ID. Got " \
                  f"{self.id_list_val}, but target does not fit the required format."
            raise FileNotFoundError(msg)

        # put the defined IDs to work
        if self.run_mode: # during training mode
            self.data_loader_cfg.id_list = training_ids

            # validation IDs
            if not self.data_loader_val_cfg is None:
                self.data_loader_val_cfg.id_list = validation_ids
                self._logger.debug(f"Validation IDs: {validation_ids}")
            # Handle the special options
            if self.validate_on_testing_set:
                self._logger.debug(f"validate_on_testing_set mode")
                self.data_loader_val_cfg.id_list = testing_ids
            if self.validate_on_training_set:
                self._logger.debug(f"validate_on_training_set mode")
                self.data_loader_val_cfg.id_list = training_ids


        else: # during inference mode
            self.data_loader_cfg.id_list = testing_ids
            if self.inference_on_training_set:
                if training_ids is None:
                    msg = "Set inference on training set option but training IDs were not specified in the template."
                    raise ArithmeticError(msg)
                self.data_loader_cfg.id_list = training_ids
            if self.inference_on_validation_set:
                if validation_ids is None:
                    msg = "Set inference on validation set option but validation IDs were not specified in the " \
                          "template."
                    raise ArithmeticError(msg)
                self.data_loader_cfg.id_list = validation_ids

        # put in the rest of the required attributes
        self.solver_cfg.cp_save_dir = self.cp_save_dir
        self.solver_cfg.cp_load_dir = self.cp_load_dir

        # select precision
        if int(torch.__version__.split('.')[0]) >= 2:
            if not self.matmul_precision in ('medium', 'high', 'highest'):
                msg = f"Precision option can only be: ['medium'|'high'|'higest]. Got {self.matmul_precision} instead."
                self._logger.warning(msg)
                return
            self._logger.info(f"Seting precision to {self.matmul_precision}")
            torch.set_float32_matmul_precision(self.matmul_precision)

    def _override_subcfg(self,
                         new_value_dict: dict,
                         target_cfg: PMIBaseCFG) -> None:
        r"""Overrides the attributes in target configuration class instance. Mainly written for solvers and data loaders
        cfg overrides.

        Args:
            new_value_dict (dict):
                Dictionary that contains key-value pairs where the key specify the attribute to override and the value
                will be the new value of that attribute in ``target_cfg``.
            target_cfg (dataclass):
                The target cfg to override.

        """
        if not isinstance(target_cfg, (PMIDataLoaderBaseCFG, SolverBaseCFG, PMIControllerCFG)):
            msg = f"Expect input `target_cfg` to be a dataclass instance, got {type(target_cfg)} instead."
            raise TypeError(msg)

        for  k, v in new_value_dict.items():
            if hasattr(target_cfg, k) and not v is None: # Dont override if new value is ``None``
                self._logger.debug(f"Overriding tag {k}: {getattr(target_cfg, k)} -> {v}")
            else:
                self._logger.debug(f"Adding new tag {k}: {v}")
            setattr(target_cfg, k, v)

    @property
    def data_loader_cfg(self):
        return self.cfg.data_loader_cfg

    @data_loader_cfg.setter
    def data_loader_cfg(self, x):
        self._logger.warning("Overriding `data_loader_cfg` with {x}.")
        self.cfg.data_loader_cfg = x

    @property
    def data_loader_val_cfg(self):
        return self.cfg.data_loader_val_cfg

    @data_loader_val_cfg.setter
    def data_loader_val_cfg(self, x):
        self._logger.warning("Overriding `data_loader_val_cfg` with {x}.")
        self.cfg.data_loader_val_cfg = x

    @property
    def solver_cfg(self) -> SolverBaseCFG:
        return self.cfg.solver_cfg

    @solver_cfg.setter
    def solver_cfg(self, x):
        self._logger.warning("Overriding `solver_cfg` with {x}.")
        self.cfg.solver_cfg = x

    def _cleanup_DDP(self):
        r""""""
        dist.destroy_process_group()

    def exec(self):
        r"""This method executes the training or inference pipeline according to the configuration. It will invoke
        :meth:`.train` or :meth:`inference` based on the flag `run_mode`.
        """
        try:
            self._logger.info(f"Before preprocess flag {self.solver_cls = }")
            self._pre_process_flags()
            self._logger.info(f"After {self.solver_cls = }")

            # write down the configurations before execution
            controller_config = {'cfg/controller/' + k: v for k, v in self.__dict__.items() if
                                 isinstance(v, (str, int, float))}
            # record the run parameters
            if self.plotting:
                self._plotter.save_dict(controller_config)
                if hasattr(self, 'guild_dict'):
                    guild_config = {'cfg/guild/' + k: v for k, v in self.guild_dict.items()}
                    self._plotter.save_dict(guild_config)
                if hasattr(self, 'guild_yaml_path'):
                    self._plotter.save_file('cfg/guild/guild.yaml', str(self.guild_yaml_path))

            # Run train or inference
            if self.run_mode:
                self.train()
            else:
                self.inference()
            return 0
        except Exception as e:
            self._logger.error("Execution failed!")
            self._logger.exception(e)
            return 1

    def train(self):
        r"""Initiate training. This is usually called automatically using :func:`.exec`. """
        #-------------------------------
        # Create training solver and net
        self._logger.info("Start training...")
        self.check_flags_sanity()

        # alter some of the flags if DDP is initialized because soem of the items can be shared
        if dist.is_initialized():
            if dist.get_rank() != 0:
                self.solver_cfg.plotting = False
        self._logger.debug(f"{self.solver_cls = }")
        solver = self.solver_cls(self.solver_cfg)
        self._logger.info("Loading train data...")
        loader = self.data_loader_cls(self.data_loader_cfg)
        self._logger.info("Loading validation data...")
        loader_val = self.data_loader_val_cls(self.data_loader_val_cfg)

        # if DDP is initialized, world size and rank are get from env variable.
        if dist.is_initialized():
            # wrap loader and solver with DDP wrappers
            self._logger.info("DDP is initiated, wrapping solver and data loader...")
            loader = PMIDistributedDataWrapper(loader, dist.get_world_size(), dist.get_rank())
            solver = SolverDDPWrapper(solver, world_size=dist.get_world_size(), rank=dist.get_rank())

        # Push dataloader to solver
        self.solver = solver
        self.solver.set_plotter(self._plotter) # This could be None
        self.solver.plotting = self.plotting
        self.solver.plotter_type = self.plotter_type
        self.solver.set_data_loader(loader, loader_val)
        self.solver.fit(self.cp_save_dir,
                        debug_validation=self.debug_validation) # TODO: move checkpoint_save argument to else where

    def inference(self):
        r"""Initiate inference. This is usually called automatically using :func:`.exec`. """
        self._logger.info("Starting evalution...")
        # Create inferencer
        self.inferencer = inferencer = self.inferencer_cls(self.solver_cfg)
        # Create dataloader
        loader = self.data_loader_cls(self.data_loader_cfg)
        inferencer.set_data_loader(loader)
        inferencer.set_plotter(self._plotter)

        if self.output_dir is not None:
            inferencer.output_dir = self.output_dir # override this flag
        else:
            self._logger.warning("Output directory is not specified. This could be an error.")

        with torch.no_grad():
            if self.inference_all_checkpoints:
                try:
                    #TODO: need to fix the output_dir for this
                    inferencer.write_out_allcps()
                except AttributeError:
                    logger.warning("Falling back to normal inference.")
                    inferencer.write_out()
            inferencer.write_out()

        # Output summary of results if implemented
        if not hasattr(inferencer, 'display_summary'):
            self._logger.info(f"No summary for the class: {inferencer.__class__.__name__}")
        try:
            inferencer.display_summary()
        except AttributeError as e:
            self._logger.exception("Error when computing summary.")

    def create_plotter(self):
        r"""Create the tensorboard plotter. Note that th """
        try:
            if self.plotter_type == 'tensorboard':
                # for legacy purpose, this has always been specified by global env variable.
                tensorboard_rootdir =  Path(os.environ.get('TENSORBOARD_LOGDIR', '/media/storage/PytorchRuns'))
                if not tensorboard_rootdir.is_dir():
                    self._logger.warning("Cannot read from TENORBOARD_LOGDIR, retreating to default path...")
                    tensorboard_rootdir = Path("/media/storage/PytorchRuns")

                # Strip the parenthesis and comma from the net name to avoid conflicts with system
                net_name = str(self.net_name) + '_' + time.strftime('%Y-%b-%d_%H%M%p', time.localtime())
                net_name = net_name.translate(str.maketrans('(),.','[]--'," "))
                self._logger.info("Creating TB writer, writing to directory: {}".format(tensorboard_rootdir))

                # create new directory
                idx = 0
                tensor_dir = tensorboard_rootdir / net_name
                while tensor_dir.is_dir():
                    tensor_dir = tensorboard_rootdir / f"{net_name}-{idx:02d}"
                    idx += 1

                writer = SummaryWriter(str(tensor_dir))
                self._plotter = TB_plotter(writer)
            elif self.plotter_type == 'neptune':
                self._logger.info("Using Neptune plotter")
                self._plotter = NP_Plotter()
                # check if there's already a run with the same guid ID
                if self.neptune_id is not None:
                    self._logger.warning(f"Neptune ID {self.neptune_id} exist, continuing plotting to this run.")
                    self._plotter.continue_run(neptune_run_id=self.neptune_id)
                else:
                    self._plotter.init_run(init_meta=self.plotter_init_meta)
        except Exception as e:
            self._logger.warning("Plotter creation encounters failure, falling back to no writer.")
            self._logger.exception(e)
            raise e
            self._plotter = None