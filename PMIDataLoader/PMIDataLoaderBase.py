import re
import logging
import configparser
from abc import *

class PMIDataLoaderBase(object):
    """
    This is the base class to allow automatic loading from main.py. All custom class should inherit this class such
    and implement the abstract methods for the main file to function properly.

    Attributes:
        debug (bool, Optional):
            Enable debug mode when loading data. Default to False.
        verbose (bool, Optional):
            Enable verbose output during data loading. Default to False.
        logging (logging, Optional):
            Specify logger. If None, default logger `__main__` would be used.
        _data_type (str):
            Name of the data type.
        _input_dir (str):
            Root directory of input to network.
        _target_dir (str, Optional):
            Root directory of input to loss function.

    Args:
        prop_dict (dict or str):

    .. note::
        * The private attributes are defined in :func:`PMIDataLoaderBase._read_params`.
        * :obj:`prob_dict` should either be directory to an ini file or a `configparser.ConfigParser` object. This
          class read from the section `[General]`. The ini file should at least consist of attribute `run_mode`. The
          child class would read from the section `[LoaderParams]` to obtain the necessary tags.


    """
    def __init__(self, prop_dict, debug=False, verbose=True, logger=None, **kwargs):
        self._prop_dict = prop_dict
        self._logger = logger if logger is None else logging.getLogger('__main__')
        self._verbose = verbose
        self._debug = debug
        if not self._check_input:
            raise AttributeError

        self._read_params(prop_dict)
        self._run_mode = self._prop_dict['General'].get('run_mode')

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
    def _load_data_set_loss_func_gt(self):
        """Inhereit in child class. Private method called to load loss function arguments. You can design the
        function to retun `None` if loss function doesn't require gt, but you stil need to inherit this.
        """
        raise NotImplemented

    @abstractmethod
    def _load_data_set_inference(self):
        """Inherit in child class. Private method called to load data for inference mode."""
        raise NotImplementedError

    def load_data_set(self):
        """
        Called in solver or inferencer to load arguments for the network training or actual inference. Normally you
        would need to inherit this but you can do so to added some custom features.
        """
        if re.match('(?=.*train.*)', self._run_mode):
            return self._load_data_set_training()
        else:
            return self._load_data_set_inference()

    def write_log(self, msg, level=logging.INFO):
        """Write log to logger if there's one"""
        if not self._logger is None and isinstance(self._logger, logging.getLoggerClass()):
            self._logger.log(self.__class__.__name__ + ": " + msg, level)
        if self._verbose:
            print(msg)

    def _read_params(self, config_file=None):
        """
        Read params from prop_dict, adds to attribute of the object. If config file is specified, every will be
        compied to the `self._loader_params`. Attributes are added as follow:

        Args:
            config_file (str or dict, Optional):
                If it a `str` point to an .ini file, it will be read and converted to a dictionary. Store to the
                attribute `self._loader_params`. The section 'LoaderParams' must exist for .ini file reading.
                Default to `None`.

        """
        self._datatype = self._prop_dict['data_type']
        self._input_dir = self._prop_dict['input_dir']
        self._target_dir = self.get_from_prop_dict('target_dir', None)

        if not config_file is None:
            if isinstance(config_file, str):
                config = configparser.ConfigParser()
                config.read_file(config_file)
                self._loader_params = dict(config['LoaderParams'])
            else:
                self._loader_params = config_file


    def get_from_prop_dict(self, key, default_value=None, tar_dict=None):
        """
        Method for convenient value extraction. Read from self._prop_dict or specified dictionary with default
        parameters if the key doesn't exist.

        Args:
            key (obj): Key to read from self.
            default_value (Optional): Value to fill in if the key is not found in `tar_dict`. Default to `None`.
            tar_dict (dict): Target dictionary. Default to `self._prop_dict`
        """
        tar_dict = self._prop_dict if tar_dict is None else tar_dict
        return tar_dict[key] if key in tar_dict else default_value