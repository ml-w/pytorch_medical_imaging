import os
import re
import configparser
import pandas as pd
import pprint
import itertools
import copy
from abc import *
from pathlib import Path
from torch.utils.data import DataLoader

import torchio as tio
from ..pmi_base_cfg import PMIBaseCFG
from .augmenter_factory import create_transform_compose
from ..med_img_dataset.PMIDataBase import PMIDataBase
from mnts.mnts_logger import MNTSLogger
from typing import *

__all__ = ['PMIDataLoaderBaseCFG', 'PMIDataLoaderBase']

class PMIDataLoaderBaseCFG(PMIBaseCFG):
    """Config required to initialize :class:`PMIDataLoader`.

    Class Attributes:
        input_dir (str or list):
            Directory of input data root dir. The directory(ies) should contain all the data that are to be
            loaded. Default to be "", which will trigger an exception if not configured.
        target_dir (str or list, Optional):
            Directory of target data that will be treated as the ground-truth. Usually required during training but not
            during inference, so this was made optional. Default to "".
        output_dir (str or list, Optional):
            The directory to deposit the output files, if any. Usually used during inference mode.
        id_globber (str, Optional):
            Regex or wildcard patter to glob the ID from the file names of the data. Each data point should be uniquely
            identified by an ID that is globbed using this pattern. The IDs are also used to match the data loaded from
            ``input_dir`` and the labels loaded from the ``target_dir``. Default to "(^[a-zA-Z0-9]+)".
        id_list (str or list, Optional):
            If only part of the data are to be loaded, use this option. Either specify a .ini file with that contains
            a section [FileList] and attributes 'training' and 'testing' with comma separated string of IDs, or a .txt
            file with each line as an ID to be loaded. If this is a list, all element of this list should be strings
            corresponding to the IDs desired to be loaded. Default to "".
        id_exclude (str, Optional)
            If you want a few IDs to be excluded for whatever reason, use this as the ultimate override. Only string is
            supported and it should be comma separated values of IDs (no space). Default to "".
        run_mode (str, Optional):
            Either 'train' or 'inference'. Default to 'train'
        debug_mode (bool, Optional):
            Debug mode flag, not used in this base class but can be passed to the child classes. Default to ``False``.

    .. hint::
        Inherit this base class to define more attributes to be used by the solvers.

    .. note::
        Please note that when using data classes with controller, the controller `id_list` attribute will override the
        :attr:`PMIDataLoaderBaseCFG.id_list` of this class.

    """
    input_dir    : str = ""
    target_dir   : str = ""
    output_dir   : str = ""
    id_list      : str = ""
    id_exclude   : Optional[str] = ""
    id_globber   : Optional[str] = "(^[a-zA-Z0-9]+)"
    run_mode     : Optional[str] = 'train'
    debug_mode   : Optional[bool] = False

    def _as_dict(self):
        r"""This function is not supposed to be private, but it needs the private tag to be spared by :func:`.__init__`
        """
        return self.__dict__

    def __str__(self):
        _d = {k: v for k, v in self.__dict__.items() if k[0] != '_'}
        if 'id_list' in _d:
            if isinstance(_d['id_list'], (list, tuple)):
                _d['id_list'] = ', '.join(_d['id_list'])
        return pprint.pformat(_d, indent=2)

    def __copy__(self):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        new_obj.__dict__.update(self.__dict__)
        return new_obj


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

    .. hint::
        The loaders can be created using both the CFG class and CFG instance. If you are unsure what their differences
        are, it is recommend to always using CFG() to create a configuration instance.

    Example
    ^^^^^^^

    **Changing the default values**

    >>> from pytorch_med_imaging.data_loader import PMIDataLoaderBaseCFG as BCFG
    >>> BCFG.input_dir = '/new/default'
    >>> cfg = BCFG()
    >>> loader = LoaderClass(cfg)

    **Override default values**

    >>> cfg = BCFG(input_dir='/override/dir')
    >>> loader = LoaderClass(cfg)


    """
    cfg_cls = PMIDataLoaderBaseCFG
    def __init__(self,
                 cfg: PMIDataLoaderBaseCFG,
                 **kwargs):
        self._cfg = cfg
        self._logger = MNTSLogger[self.__class__.__name__]

        if not self._check_input:
            raise AttributeError

        self._logger.info("Data loader was configured with options: {}".format(str(cfg)))
        self._read_config(cfg)

    @abstractmethod
    def _check_input(self):
        """Inherit in child class. The input dict should be checked in this function."""
        raise NotImplementedError

    @abstractmethod
    def _load_data_set_training(self) -> tio.Queue:
        """Inherit in child class. Private method called for training mode. Returns whatever goes into the network
        as list. Validation set loading should have identical settings
        """
        raise NotImplementedError

    @abstractmethod
    def _load_data_set_inference(self) -> tio.Queue:
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
        if self.run_mode:
            return self._load_data_set_training(exclude_augment = False if exclude_augment is None else exclude_augment)
        else:
            return self._load_data_set_inference()

    def get_torch_data_loader(self,
                              batch_size: int,
                              exclude_augment: bool = False):
        r"""Use this function to obtain the dataloader based on the recommend torchio settings.

        Args:
            batch_size (int):
                Batch size.
            exclude_augment (bool, Optional):
                Pass to :meth:`._load_data_set_training` or :meth:`._load_data_set_inference`

        Returns:
            torch.utils.DataLoader
        """
        # num_workers = 0 if self.sampler is not None else os.cpu_count() - 2
        num_workers = 0
        self._logger.debug(f"Creating torch data loader with {num_workers} workers.")
        if self.run_mode: # training
            _iterable = self._load_data_set_training(exclude_augment)
            _shuffle = True if isinstance(_iterable, tio.SubjectsDataset) else not _iterable.shuffle_subjects
            # _shuffle = False # debug
            out_loader = DataLoader(_iterable,
                                    batch_size  = batch_size,
                                    # if tio.Queue have already shuffled, don't do added shuffling because it takes time
                                    shuffle     = _shuffle,
                                    num_workers = num_workers,
                                    drop_last   = True,
                                    pin_memory  = False)
        else:
            out_loader = DataLoader(self._load_data_set_inference(),
                                    batch_size  = batch_size,
                                    shuffle     = False,
                                    num_workers = num_workers,
                                    drop_last   = False,
                                    pin_memory  = False)
        return out_loader

    def _read_config(self, config_file=None):
        """
        Read params from prop_dict, adds to attribute of the object. If config file is specified, every will be
        copied to the `self.__dict__`. See the CFG class for more. If this function is called without arguments, it will
        try to locate the :attr:`cfg` and load configurations from it. If it still can't find it, it will finally turn
        to the class attribute ``cfg_cls``, which is defined for each data loader types, to load the basic setting.

        Args:
            config_file (str or dict, Optional):
                If it a `str` point to an .ini file, it will be read and converted to a dictionary. Store to the
                attribute `self._loader_params`. The section 'LoaderParams' must exist for .ini file reading.
                Default to `None`.

        .. note::
            Most of the time the basic setting is not enough to properly load the data. Therefore, make sure you have
            the configurations properly set before calling :func:`get_torch_data_loader`.

        """
        # Loading basic inputs
        if config_file is None:
            # Load default if cfg was not set properly
            cls = getattr(self, 'cfg', self.cfg_cls())
        else:
            cls = config_file
            
        cls_dict = { attr: getattr(cls, attr) for attr in dir(cls) }
        self.__dict__.update(cls_dict)

        if isinstance(self.run_mode, str):
            if re.match('(?=.*train.*)', self.run_mode):
                self.run_mode = True
            else:
                self.run_mode = False

        # load to ``self.id_list``
        self._read_id_configs()

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
                    self.transform = None
            else:
                self._logger.warning(f"Transform file provided but could not be located! Got {str(self.augmentation)}")
                self.augmentation = False
                self.transform = None
        else:
            self._logger.warning(f"`self.augmentation` was not defined!")
            self.augmentation = False
            self.transform = None
        return self.transform

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
                directly remove these ids from ``self.id_list``.

        .. Note::
            The default values of ``id_list`` and ``id_exclude`` are both empty strings ''. If they are empty strings
            no id filtering will be executed.
        """
        if not self.id_list in ("", None) and not isinstance(self.id_list, list):
            if self.id_list.endswith('.ini'):
                self.id_list = self.parse_ini_filelist(self.id_list, self.run_mode)
            elif self.id_list.endswith('.txt'):
                self.id_list = [r.rstrip() for r in open(self.id_list).readlines()]
            else:
                if self.id_list.find(',') >= 0:
                    self.id_list = self.id_list.split(',')
            self.id_list.sort()
        elif isinstance(self.id_list, list):
            if not all([isinstance(ll, str) for ll in self.id_list]):
                types = [type(ll) for ll in self.id_list]
                msg = f"Expect all components in `id_list` to be string, got: " \
                      f"{[(a, b) for a, b in zip(self.idlist, types)]}"
                raise TypeError(msg)
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
