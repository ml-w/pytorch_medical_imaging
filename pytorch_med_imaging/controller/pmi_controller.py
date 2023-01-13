import re

from ..solvers import SolverBaseCFG, SolverDDPWrapper
from ..pmi_base_cfg import PMIBaseCFG
from ..pmi_data_loader import PMIDataLoaderBase, PMIDataLoaderBaseCFG, PMIDistributedDataWrapper
from pathlib import Path
from typing import Union, Optional

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
            ``data_loader_val_cfg``.
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
        plot_tb (bool, Optional):
            Default to ``True``.
        log_dir (str, Optional):
            Default directory for outputting the log file. Default to ``'./Backup/Log'``.
        keep_log (bool, Optional):
            Default to ``True``.
        verbose (bool, Optional):
            Defaul to ``True``.

    .. note::
        Don't confuse the `id_list` in this CFG with that in :class:`SolverBase`, the later is more flexible and can
        accept various specification formats

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
    plot_tb                    : Optional[bool] = True # if false no plots


    # log related
    log_dir : Optional[str]  = './Backup/Log/'
    keep_log: Optional[bool] = True
    verbose : Optional[bool] = True

    # Configurations
    data_loader_cfg    : PMIDataLoaderBaseCFG           = None
    data_loader_val_cfg: Optional[PMIDataLoaderBaseCFG] = None
    solver_cfg         : SolverBaseCFG                  = None
    data_loader_cls    : type = None
    data_loader_val_cls: type = None
    solver_cls         : type = None
    inferencer_cls     : type = None


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
            cls_dict = { attr: getattr(cls, attr) for attr in dir(cls) }
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
            (self.inferencer_cls, type)
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
        else:
            msg = f"Suffix of the override file should be align with guildai's definitions. Currently only '.ini', " \
                  f"'.yaml' and '.json' were implemented. Got {override_file.name} instead."
            raise ArithmeticError(msg)

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
                (self, 'id_list')
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
                if isinstance(self.id_list_val, (str, Path)):
                    with open(self.id_list_val, 'r') as _val_txt:
                        validation_ids = [r.rstrip() for r in _val_txt.readlines()]
                    validation_ids.sort()
                else:
                    validation_ids = self.id_list_val
            else:
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
        self._pre_process_flags()

        # Run train or inference
        if self.run_mode:
            self.train()
        else:
            self.inference()

    def train(self):
        r"""Initiate training. This is usually called automatically using :func:`.exec`. """
        #-------------------------------
        # Create training solver and net
        self._logger.info("Start training...")
        self.check_flags_sanity()

        # alter some of the flags if DDP is initialized because soem of the items can be shared
        if dist.is_initialized():
            if dist.get_rank() != 0:
                self.solver_cfg.plot_to_tb = False

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
        self.solver.set_data_loader(loader, loader_val)
        self.solver.fit(self.cp_save_dir,
                        debug_validation=self.debug_validation) # TODO: move checkpoint_save argument to else where

    def inference(self):
        r"""Initiate inference. This is usually called automatically using :func:`.exec`. """
        self._logger.info("Starting evalution...")
        self.inferencer = inferencer = self.inferencer_cls(self.solver_cfg)
        loader = self.data_loader_cls(self.data_loader_cfg)
        inferencer.set_data_loader(loader)
        inferencer.output_dir = self.output_dir # override this flag

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

