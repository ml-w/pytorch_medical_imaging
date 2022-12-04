import os
import re
import configparser
import pandas as pd
import itertools
from abc import *
from pathlib import Path

import torchio as tio
from .augmenter_factory import create_transform_compose
from ..med_img_dataset.PMIDataBase import PMIDataBase
from mnts.mnts_logger import MNTSLogger

class PMIDataLoaderBaseCFG:
    """Config required to initialize PMIDataLoader"""
    input_dir    : str = ""
    target_dir   : str = ""
    output_dir   : str = ""
    id_list      : str = ""
    id_exclude   : str = ""
    id_globber   : str = "(^[a-zA-Z0-9]+)"
    run_mode     : str = "" # 'train' or 'inference'


class PMIDataLoaderBase(object):
    """
    This is the base class to allow automatic loading from main.py. All custom class should inherit this class such
    and implement the abstract methods for the main file to function properly.

    Attributes:
        data_type (str):
            Name of the data type.
        input_dir (str):
            Root directory of input to network.
        target_dir (str, Optional):
            Root directory of input to loss function.

    Args:
        cfg (dict or str):
            Either a dictionary of parameters or a str pointing to an .ini file that hold the required parameters.
        run_mode (str):
            {'train', 'inference'}. Decide the behavior of objects.
        debug (bool, Optional):
            Enable debug mode when loading data. Default to `False`.
        verbose (bool, Optional):
            Enable verbose output during data loading. Default to `False`.
        logging (logging, Optional):
            Specify logger. If `None`, default logger `__main__` would be used.



    .. note::
        * The private attributes are defined in :func:`PMIDataLoaderBase._read_config`.
        * :obj:`prob_dict` should either be directory to an ini file or a `configparser.ConfigParser` object. This
          class read from the section `[General]`. The ini file should at least consist of attribute `run_mode`. The
          child class would read from the section `[LoaderParams]` to obtain the necessary tags.


    """
    def __init__(self,
                 cfg: PMIDataLoaderBaseCFG,
                 run_mode='training', debug=False, verbose=True, logger=None, **kwargs):
        self._cfg = cfg
        self._logger = logger if not logger is  None else MNTSLogger[self.__class__.__name__]
        self._verbose = verbose
        self._debug = debug
        self._run_mode = run_mode

        if isinstance(self._run_mode, str):
            if re.match('(?=.*train.*)', self._run_mode):
                self._run_mode = True
            else:
                self._run_mode = False

        if not self._check_input:
            raise AttributeError

        self._read_config(cfg)

    @abstractmethod
    def _check_input(self):
        """Inherit in child class. The input dict should be checked in this function."""
        raise NotImplementedError

    @abstractmethod
    def _load_data_set_training(self):
        """Inherit in child class. Private method called for training mode. Returns whatever goes into the network
        as list. Validation set loading should have identical settings
        """
        raise NotImplementedError

    @abstractmethod
    def _load_data_set_inference(self):
        """Inherit in child class. Private method called to load data for inference mode."""
        raise NotImplementedError

    @staticmethod
    def parse_ini_filelist(filelist, mode):
        r"""
        Parse the ini file for this class.

        Args:
            filelist (str): Relative directory to the ini filelist.

        Returns:
            (list): A list containing the IDs specifed in the target file list.

        Examples:

            An example of .ini file list should look something like this,

            file_list.ini::

                [FileList]
                testing=ID_0,ID_1,...,ID_n
                training=ID_a,ID_b,...,ID_m

        """
        assert os.path.isfile(filelist), "Cannot locate filelist {}".format(filelist)

        fparser = configparser.ConfigParser()
        fparser.read(filelist)

        # test
        if re.match('(?=.*train.*)', mode) is not None:
            return fparser['FileList'].get('training').split(',')
        else:
            return fparser['FileList'].get('testing').split(',')

    def _load_default_attr(self, default_dict):
        r"""
        Load default dictionary as attr from loader params
        """
        final_dict = {}
        for key in default_dict:
            val = default_dict[key]
            if isinstance(val, bool):
                final_dict[key] = self.get_from_loader_params_with_boolean(key, default_dict[key])
            elif isinstance(val, str):
                final_dict[key] = self.get_from_loader_params(key, default_dict[key])
            else:
                final_dict[key] = self.get_from_loader_params_with_eval(key, default_dict[key])
        self._logger.debug(f"final_dict: {final_dict}")
        self.__dict__.update(final_dict)

    def load_dataset(self, exclude_augment = None):
        r"""
        Called in solver or inferencer to load arguments for the network training or actual inference. Normally you
        would need not to inherit this but you can do so to added some custom features.

        Args:
            exclude_augment (bool, Optional):
                Use this to pass the option to load_dataset

        Returns:
            (ImageDataSet or [ImageDataSet, ImageDataSet]): Depend on whether `self._run_mode` is `training` or
            `inference`, return dataset for solver or inferencer to process.


        """
        if self._run_mode:
            return self._load_data_set_training(exclude_augment = False if exclude_augment is None else exclude_augment)
        else:
            return self._load_data_set_inference()

    def write_log(self, msg, level=MNTSLogger.INFO):
        """Write log to logger if there's one"""
        if not self._logger is None and isinstance(self._logger, MNTSLogger):
            self._logger.log(self.__class__.__name__ + ": " + msg, level)
        if self._verbose:
            print(msg)

    def _read_config(self, config_file=None):
        """
        Read params from prop_dict, adds to attribute of the object. If config file is specified, every will be
        compied to the `self._loader_params`. Attributes are added as follow:

        Args:
            config_file (str or dict, Optional):
                If it a `str` point to an .ini file, it will be read and converted to a dictionary. Store to the
                attribute `self._loader_params`. The section 'LoaderParams' must exist for .ini file reading.
                Default to `None`.

        """
        # Loading basic inputs
        if not config_file is None:
            if isinstance(config_file, type):
                cls = config_file
            else:
                cls = config_file.__class__
            cls_dict = { attr: getattr(cls, attr) for attr in dir(cls) }
            self.__dict__.update(cls_dict)

        # load to ``self.id_list``
        self._read_id_configs()

    def get_from_config(self, section, key, default_value=None, tar_dict=None):
        """
        Method for convenient value extraction. Read from self._prop_dict or specified dictionary with default
        parameters if the key doesn't exist.

        Args:
            key (obj): Key to read from self.
            default_value (Optional): Value to fill in if the key is not found in `tar_dict`. Default to `None`.
            tar_dict (dict): Target dictionary. Default to `self._prop_dict`
        """
        tar_dict = self._cfg if tar_dict is None else tar_dict
        try:
            out = tar_dict[section][key]
            return out
        except:
            return default_value

    def get_from_config_with_eval(self, section, key, default_value=None, tar_dict=None):
        """
        Method for convenient value extraction. Read from self._prop_dict or specified dictionary with default
        parameters if the key doesn't exist.

        If the target is string, `eval()` is called to convert them to python objects.

        Args:
            key (obj): Key to read from self.
            default_value (Optional): Value to fill in if the key is not found in `tar_dict`. Default to `None`.
            tar_dict (dict): Target dictionary. Default to `self._prop_dict`
        """
        tar_dict = self._cfg if tar_dict is None else tar_dict
        try:
            out = tar_dict[section][key]
            if isinstance(out, str):
                try:
                    out = eval(out)
                except:
                    self._logger.warning("Failed when trying to evaluate {}.".format(key))
                    raise ValueError("Cannot evaluate target key.")
            return out
        except:
            return default_value

    def get_from_config_with_boolean(self, section, key, default_value=None, tar_dict=None):
        """
        Same as :func:`get_from_loader_params` with getboolean().

        Args:
            key (obj): Key to read from self.
            default_value (Optional): Value to fill in if the key is not found in `tar_dict`. Default to `None`.
            tar_dict (dict): Target dictionary. Default to `self._prop_dict`
        """
        tar_dict = self._cfg if tar_dict is None else tar_dict
        try:
            out = tar_dict[section].getboolean(key)
            return out
        except:
            self._logger.warning("Cannot getboolean from target section {} with key {}".format(section, key))
            return default_value

    def get_from_loader_params(self, key, default_value=None):
        """
        Method for convenient value extraction. Read from self._prop_dict[LoaderParams] with default
        parameters if the key doesn't exist.

        Args:
            key (obj): Key to read from self.
            default_value (Optional): Value to fill in if the key is not found in `tar_dict`. Default to `None`.
            tar_dict (dict): Target dictionary. Default to `self._prop_dict`
        """
        try:
            tar_dict = self._cfg['LoaderParams']
            out = tar_dict[key]
            return out
        except:
            return default_value

    def get_from_loader_params_with_eval(self, key, default_value=None):
        """
        Same as :func:`get_from_loader_params` with eval().

        Args:
            key (obj): Key to read from self.
            default_value (Optional): Value to fill in if the key is not found in `tar_dict`. Default to `None`.
            tar_dict (dict): Target dictionary. Default to `self._prop_dict`
        """
        try:
            tar_dict = self._cfg['LoaderParams']
            out = tar_dict[key]
            if isinstance(out, str):
                try:
                    out = eval(out)
                except:
                    self._logger.warning("Failed when trying to evaluate {}.".format(key))
                    raise ValueError("Cannot evalute target key.")
            return out
        except:
            return default_value

    def get_from_loader_params_with_boolean(self, key, default_value=None):
        """
        Same as :func:`get_from_loader_params` with getboolean().

        Args:
            key (obj): Key to read from self.
            default_value (Optional): Value to fill in if the key is not found in `tar_dict`. Default to `None`.
            tar_dict (dict): Target dictionary. Default to `self._prop_dict`
        """
        try:
            tar_dict = self._cfg['LoaderParams']
            try:
                out = tar_dict.getboolean(key, default_value)
            except:
                self._logger.warning('Cannot get boolean value from key {}'.format(key))
                raise ValueError("Cannot get boolean value from key {}".format(key))
            return out
        except:
            return default_value

    def get_target_attributes(self, section, tar_keys, tar_def_values=None, tar_eval_flag=None, tar_dict=None):
        """
        Get attributes from target dictionary with default values in batch.

        Args:
            tar_keys (list of str):
            tar_def_values (list of values, Optional):
            tar_eval_flat (list of bool):
            tar_dict (dict): Target dictionary. Default to `self._prop_dict`

        Returns:
            out_dict (dict): Dictionary with
        """
        import warnings
        warnings.deprecation()

        out_dict = {}
        for k, default_value, eval_flag in zip(tar_keys, tar_def_values, tar_eval_flag):
            if eval_flag:
                _func = self.get_from_config_with_eval
            else:
                _func = self.get_from_config
            out_dict[k] = _func(section, k, default_value=default_value, tar_dict=tar_dict)
        return out_dict

    def _create_transform(self, exclude_augment = False):
        r"""Wrapper function of ``create_transform_compose()``. This creates a ``tio.Compose``

        Args:
            exclude_augment (bool):
                If ``True``, only normalization transform will be built. See also :func:`create_transform_compose`.

        Required attributes:
            augmentation (str):
                Path to the yaml file that defines the tio.Compose. If this is `None`, this function will also return
                `None`.

        Returns:
            compose (tio.Compose):
        """
        if isinstance(self.augmentation, str):
            if Path(self.augmentation).is_file():
                try:
                    self.transform = create_transform_compose(self.augmentation, exclude_augment=exclude_augment)
                    self._logger.debug(f"Built transform: {self.transform}")
                except Exception as e:
                    self._logger.error(f"Failed to create augmentation from file: {self.augmentation}. Got {e}")
                    self.augmentation = False
            else:
                self._logger.warning(f"Transform file provided but could not be located! Got {str(self.augmentation)}")
            return self.transform
        else:
            self._logger.warning(f"`self.augmentation` was not defined!")
            return None

    def _pack_data_into_subjects(self,
                                 data_dict: dict,
                                 transform: tio.Transform = None) -> tio.SubjectsDataset:
        r"""Create subjects from a dictionary of data"""
        data_exclude_none = {k: v for k, v in data_dict.items() if v is not None}

        # check if all items has the same length
        if not len(set([len(v) for v in data_exclude_none.values()])):
            msg = f"Expect all data to have the same length, but got: "
            msg += str({k: len(v) for k, v in data_dict.items()})
            raise IndexError(msg)

        # check if the IDs are aligned
        ids = {k: set(_d.get_unique_IDs()) for k, _d in data_dict.items() if isinstance(_d, PMIDataBase)}
        if not all([ids[a] == ids[b] for a, b in itertools.combinations(ids.keys(), 2)]):
            uni = set.union(*list(ids.values()))
            _table = pd.concat([pd.Series([index in v for index in uni], index=uni, name=k) for k, v in ids.items()], axis=1)
            _table.sort_index(inplace=True)
            _table = _table[[False in list(row[1]) for row in _table.iterrows()]]
            msg = f"Expect all data to have same unique IDs, but some are not: \n"
            msg += _table.to_string()
            raise IndexError(msg)

        subjects = [tio.Subject(**{k: v for k, v in zip(data_exclude_none.keys(), row)})
                    for row in zip(*data_exclude_none.values())]
        subjects = tio.SubjectsDataset(subjects=subjects, transform=transform)
        return subjects

    def _read_id_configs(self):
        r"""This function is called after reading the cfg file, this will sort out the ids to be loaded when creating
        the dataloaders.

        Required attributes:
            id_list (str):
                If this is an ini file, will invoke ``parse_ini_filelist()`` to get the ids. If its a .txt file, will
                read each line in the txt file into the id_list.
            id_exclude (str or list, Optional):
                If this is not ``''``, will check if it is a path or a list. If it is a path, assume it is a comma
                separated list where the elements are to be removed from ``self.id_list``. If it is already a list, will
                directly remove these ids from ``self.id_list.

        .. Note::
            The default values of ``id_list`` and ``id_exclude`` are both empty strings ''. If they are empty strings
            no id filtering will be executed.
        """
        if not self.id_list in ("", None):
            if self.id_list.endswith('.ini'):
                self.id_list = self.parse_ini_filelist(self.id_list, self._run_mode)
            elif self.id_list.endswith('.txt'):
                self.id_list = [r.rstrip() for r in open(self.id_list).readlines()]
            else:
                if self.id_list.find(',') >= 0:
                    self.id_list = self.id_list.split(',')
            self.id_list.sort()
        else:
            self.id_list = None
        self.id_exclude = self._cfg.id_exclude
        if not self.id_exclude in ("", None):
            if isinstance(self.id_exclude, str) and isinstance(self.id_list, (list, tuple)):
                self.id_exclude = self.id_exclude.split(',')
                for e in self.id_exclude:
                    if e in self.id_list:
                        self._logger.info("Removing {} from the list as specified.".format(e))
                        self.id_list.remove(e)
