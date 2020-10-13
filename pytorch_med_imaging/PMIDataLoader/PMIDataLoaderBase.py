import os
import re
import configparser
from abc import *
from ..logger import Logger

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
        prop_dict (dict or str):
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
        * The private attributes are defined in :func:`PMIDataLoaderBase._read_params`.
        * :obj:`prob_dict` should either be directory to an ini file or a `configparser.ConfigParser` object. This
          class read from the section `[General]`. The ini file should at least consist of attribute `run_mode`. The
          child class would read from the section `[LoaderParams]` to obtain the necessary tags.


    """
    def __init__(self, prop_dict, run_mode='training', debug=False, verbose=True, logger=None, **kwargs):
        self._prop_dict = prop_dict
        self._logger = logger if not logger is  None else Logger[self.__class__.__name__]
        self._verbose = verbose
        self._debug = debug
        self._run_mode = run_mode
        if not self._check_input:
            raise AttributeError

        self._read_params(prop_dict)

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
        assert os.path.isfile(filelist), "Cannot locate filelist"

        fparser = configparser.ConfigParser()
        fparser.read(filelist)

        # test
        if re.match('(?=.*train.*)', mode) is not None:
            return fparser['FileList'].get('training').split(',')
        else:
            return fparser['FileList'].get('testing').split(',')

    def load_dataset(self):
        """
        Called in solver or inferencer to load arguments for the network training or actual inference. Normally you
        would need not to inherit this but you can do so to added some custom features.

        Returns:
            (ImageDataSet or [ImageDataSet, ImageDataSet]): Depend on whether `self._run_mode` is `training` or
            `inference`, return dataset for solver or inferencer to process.


        """
        if re.match('(?=.*train.*)', self._run_mode):
            return self._load_data_set_training()
        else:
            return self._load_data_set_inference()

    def write_log(self, msg, level=Logger.INFO):
        """Write log to logger if there's one"""
        if not self._logger is None and isinstance(self._logger, Logger):
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
        try:
            self._input_dir = self.get_from_config('Data', 'input_dir')
            self._target_dir = self.get_from_config('Data', 'target_dir', default_value=None)
        except IndexError as e:
            self.write_log("Can't read {} from input config".format(e))
            raise IndexError(e)

        if not config_file is None:
            if isinstance(config_file, str):
                config = configparser.ConfigParser()
                config.read_file(config_file)
                self._loader_params = dict(config['LoaderParams'])
            else:
                self._loader_params = config_file


    def get_from_config(self, section, key, default_value=None, tar_dict=None):
        """
        Method for convenient value extraction. Read from self._prop_dict or specified dictionary with default
        parameters if the key doesn't exist.

        Args:
            key (obj): Key to read from self.
            default_value (Optional): Value to fill in if the key is not found in `tar_dict`. Default to `None`.
            tar_dict (dict): Target dictionary. Default to `self._prop_dict`
        """
        tar_dict = self._prop_dict if tar_dict is None else tar_dict
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
        tar_dict = self._prop_dict if tar_dict is None else tar_dict
        try:
            out = tar_dict[section][key]
            if isinstance(out, str):
                out = eval(out)
            return out
        except:
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
            tar_dict = self._prop_dict['LoaderParams']
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
            tar_dict = self._prop_dict['LoaderParams']
            out = tar_dict[key]
            if isinstance(out, str):
                out = eval(out)
            return out
        except:
            return default_value

    def get_target_attributes(self, tar_keys, tar_def_values=None, tar_eval_flag=None, tar_dict=None):
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

        out_dict = {}
        for k, default_value, eval_flag in zip(tar_keys, tar_def_values, tar_eval_flag):
            if eval_flag:
                _func = self.get_from_loader_params_with_eval
            else:
                _func = self.get_from_config
            out_dict[k] = _func(k, default_value=default_value, tar_dict=tar_dict)