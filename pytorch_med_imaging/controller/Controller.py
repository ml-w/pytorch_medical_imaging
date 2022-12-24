from ..pmi_data_loader import PMIDataLoaderBase
from ..solvers import SolverBase
from ..inferencers import InferencerBase
from ..tb_plotter import TB_plotter
from dataclasses import dataclass, fields, _is_dataclass_instance, asdict
from pathlib import Path
from typing import Any, IO, Union, Optional

from pytorch_med_imaging.solvers import *
from pytorch_med_imaging.inferencers import *
from pytorch_med_imaging.pmi_data_loader import *

import yaml
import configparser
from mnts.mnts_logger import MNTSLogger

PathLike = Union[str, Path]

r"""
A simple API is used to override the template setting through a json

"""
@dataclass
class PMIControllerCFG:
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
        debug (bool, Optional):
            Default to ``False``
        debug_validation (bool, Optional):
            Default to ``False``
        validate_on_test_set (bool, Optional):
            Default to ``False``
        inference_on_training_set (bool, Optional):
            Default to ``False``
        inference_on_validation_set (bool, Optional):
            Default to ``False``
        inference_all_checkpoints (bool, Optional):
            Default to ``False``
        plot_tb (bool, Optional):
            Default to ``True``

    .. note::
        Don't confuse the `id_list` in this CFG with that in :class:`SolverBase`, the later is more flexible and can
        accept various specification formats

    """
    fold_code                  : str            = None
    run_mode                   : str            = 'training'
    id_list                    : PathLike       = None
    id_list_val                : PathLike       = None
    output_dir                 : PathLike       = None
    debug                      : Optional[bool] = False
    debug_validation           : Optional[bool] = False
    validate_on_test_set       : Optional[bool] = False # this changes the how the validation data loader IDs in training mode
    inference_on_training_set  : Optional[bool] = False # this changes the data loader IDs in the inference mode
    inference_on_validation_set: Optional[bool] = False
    inference_all_checkpoints  : Optional[bool] = False
    plot_tb                    : Optional[bool] = True # if false no plots


    # Configurations
    data_loader_cfg    : PMIDataLoaderBaseCFG           = None
    data_loader_val_cfg: Optional[PMIDataLoaderBaseCFG] = None
    solver_cfg         : SolverBaseCFG                  = None
    data_loader_cls    : type = None
    data_loader_val_cls: type = None
    solver_cls         : type = None
    inferencer_cls     : type = None


    def __init__(self, **kwargs):
        # load class attributes as default values of the instance attributes
        cls = self.__class__
        cls_dict = { attr: getattr(cls, attr) for attr in dir(cls) }
        for key, value in cls_dict.items():
            if key in ('solver_cls', 'inferencer_cls'):
                continue

            if not key[0] == '_':
                try:
                    setattr(self, key, value)
                except:
                    msg = f"Error when initializing: {key}: {value}"
                    raise AttributeError(msg)

        # replace instance attributes
        if len(kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __str__(self):
        _d = {k: v for k, v in self.__dict__.items() if k[0] != '_'}
        _d['net'] = self.net._get_name()
        return pprint.pformat(_d, indent=2)

    def _as_dict(self):
        r"""This function is not supposed to be private, but it needs the private tag to be spared by :func:`.__init__`
        """
        return self.__dict__

    def __iter__(self):
        cls_dict = self._get_dict()
        for k, v in cls_dict.items():
            yield k, v

class PMIController(object):
    r"""The controller to initiate training or inference. Based on the input cfg, this class will create a solver or
    an inferencer, and also the dataloaders.
    """
    def __init__(self, cfg):
        self._cfg = cfg
        self._logger = MNTSLogger[self.__class__.__name__]

        # Load configs
        self._load_config(cfg)

        # This is the root that dictates the behavior of the controller.
        if re.match('(?=.*train.*)', self.run_mode):
            self._logger.info("Controller set to training mode.")
            self.run_mode = True
        else:
            self._logger.info("Controller set to inference mode")
            self.run_mode = False

        self._required_attributes_train = [
            ''
        ]

    def _load_config(self, config_file = None):
        r"""Function to load the configurations. If ``config_file`` is ``None``, load the default class
        :class:`SolverBaseCFG` that is stored as a class attribute :attr:`cls_cfg`.
        """
        # Loading basic inputs
        if not config_file is None:
            cls = config_file
            cls_dict = { attr: getattr(cls, attr) for attr in dir(cls) }
            self.__dict__.update(cls_dict)
            self.__class__.cls_cfg = cls
            self.cfg = config_file
        else:
            self._logger.warning("_load_config called without arguments.")

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

        pass

    def override_cfg(self, override_file: PathLike) -> None:
        r"""This method is the key method to control the pipeline execution with the AI experimental manager `guild`.

        Args:
            override_file (str or Path):
                Path to a yaml file to override the cfgs.

        Example:

            example_override.ini

            ```{ini}

            ```


        """
        override_file = Path(override_file)
        if not override_file.is_file():
            msg = f"Expect input to be a directory to a yaml file, got {override_file} but it doesn't point to a file."
            raise FileNotFoundError(msg)

        if override_file.suffix in ('.yml', '.yaml'):
            override_dict = yaml.load(override_file)
        else:
            msg = f"Suffix of the override file should be align with guildai's definitions. Currently only '.ini', " \
                  f"'.yaml' and '.json' were implemented. Got {override_file.name} instead."
            raise ArithmeticError(msg)

        # Override template setting with flags
        solver_override = override_dict.get('solver_cfg', None)
        data_loader_override = override_dict.get('data_loader_cfg', None)
        data_loader_val_override = override_dict.get('data_loader_val_cfg', None)
        if not solver_override is None:
            self._override_subcfg(solver_override, self.solver_cfg)
        if not data_loader_override is None:
            self._override_subcfg(data_loader_override, self.data_loader_cfg)
        if not data_loader_val_override is None and not self.data_loader_val_cfg is None:
            self._override_subcfg(data_loader_val_override, self.data_loader_val_cfg)

    def _pre_process_flags(self) -> None:
        r"""The flags defined in :class:`PMIControllerCFG` might need to further change the CFGs of the solver and the
        data loader. This method implements this.
        """
        # Fold code replace filelist and checkpoints
        if not self.fold_code is None:
            # rebuild data loader id lists
            replace_target = [
                # (self.data_loader_cfg    , 'id_list')      # id_list for both loaders
                # (self.data_loader_val_cfg, 'id_list'),
                # (self.data_loader_cfg    , 'output_dir')   # output_dir is overrided by controller CFG
                # (self.data_loader_val_cfg, 'output_dir')   # output_dir is overrided by controller CFG
                (self.solver_cfg           , 'cp_load_dir'),
                (self.solver_cfg           , 'cp_save_dir'),
                # (self.solver_cfg         , 'output_dir') , # output_dir is overrided by controller CFG
                (self                      , 'log_dir'),
                (self                      , 'cp_load_dir'),
                (self                      , 'cp_save_dir'),
                (self                      , 'output_dir'),
                (self                      , 'id_list')
            ]
            for inst, attr in replace_target:
                _old = getattr(inst, attr, None)
                if _old is None:
                    continue
                _new = _old.replace('{fold_code}', self.fold_code)
                self._logger.debug(f"Replace {_old} with {_new}")
                setattr(inst, attr, _new)

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
            # Handle the special options
            if self.validate_on_test_set:
                self.data_loader_val_cfg.id_list = testing_ids

        else: # during inference mode
            self.data_loader_cfg.id_list = testing_ids
            if self.inference_on_training_set:
                if training_ids is None:
                    msg = "Set inference on training set option but training IDs were not specified in the template."
                    raise ArithmeticError(msg)
                self.data_loader_cfg = training_ids
            if self.inference_on_validaiton_set:
                if validation_ids is None:
                    msg = "Set inference on validation set option but validation IDs were not specified in the " \
                          "template."
                    raise ArithmeticError(msg)
                self.data_loader_cfg.id_list = validation_ids

        # don't forget validation IDs
        if not self.data_loader_val_cfg is None:
            self.data_loader_val_cfg.id_list = validation_ids

    def _override_subcfg(self,
                         new_value_dict: dict,
                         target_cfg: dataclass) -> None:
        r"""Overrides the attributes in target configuration class instance. Mainly written for solvers and data loaders
        cfg overrides.

        Args:
            new_value_dict (dict):
                Dictionary that contains key-value pairs where the key specify the attribute to override and the value
                will be the new value of that attribute in ``target_cfg``.
            target_cfg (dataclass):
                The target cfg to override.

        """
        if not _is_dataclass_instance(target_cfg):
            msg = f"Expect input `target_cfg` to be a dataclass instance, got {type(cfg)} instead."
            raise TypeError(msg)

        for  k, v in new_value_dict.items():
            if hasattr(target_cfg, k) and not v is None: # Dont override if new value is ``None``
                self._logger.debug(f"Overriding tag {k}: {getattr(target_cfg, k)} -> {v}")
            setattr(target_cfg, k, v)

    @property
    def data_loader_cfg(self):
        return self.cfg.data_loader_cfg

    @property
    def data_loader_val_cfg(self):
        return self.cfg.data_loader_val_cfg

    @property
    def solver_cfg(self) -> SolverBaseCFG:
        return self.cfg.solver_cfg

    def exec(self):
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
        self.solver = solver = self.solver_cls(self.solver_cfg)
        loader = self.data_loader_cls(self.data_loader_cfg)
        loader_val = self.data_loader_val_cls(self.data_loader_val_cfg)
        # Push dataloader to solver
        solver.set_data_loader(loader, loader_val)

        solver.fit(self.cp_save_dir,
                   debug_validation=self.debug_validation) # TODO: move checkpoint_save argument to else where

    def inference(self):
        r"""Initiate inference. This is usually called automatically using :func:`.exec`. """
        self._logger.info("Starting evalution...")
        self.inferencer = inferencer = self.inferencer_cls(self.solver_cfg)
        loader = self.data_loader_cls(self.data_loader_cfg)
        inferencer.set_data_loader(loader)
        inferencer.output_dir = self.output_dir # override this flag
        inferencer.load_checkpoint(self.cp_load_dir)

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
