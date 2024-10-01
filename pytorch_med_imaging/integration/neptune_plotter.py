import os
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Union

import neptune
import re

from mnts.mnts_logger import MNTSLogger


def check_init(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.np_run is None:
            self._logger.error("Neptune run has not been initialized. Have you run 'init_run()'?")
        return func(self, *args, **kwargs)
    return wrapper

class NP_Plotter(object):
    r"""Plotter for PMI to log information to Neptune.ai.
    """
    def __init__(self, project: str = None, api_token: str = None):
        # Create logger
        self._logger = MNTSLogger[self.__class__.__name__]

        # Attributes
        self.project = project or os.environ.get("NEPTUNE_PROJECT", None)
        self.api_token = api_token or os.environ.get("NEPTUNE_API_TOKEN", None)
        self.np_run: neptune.Run = None

        # * Create/Connect Neptune object
        self.np_project = neptune.init_project(self.project, api_token= self.api_token, mode='debug')

    @property
    def api_token(self):
        return os.environ.get('NEPTUNE_API_TOKEN', None)

    @api_token.setter
    def api_token(self, tok: str):
        os.environ['NEPTUNE_API_TOKEN'] = tok

    @property
    def project(self):
        return os.environ.get('NEPTUNE_PROJECT', None)

    @project.setter
    def project(self, val):
        os.environ['NEPTUNE_PROJECT'] = val

    def init_run(self, init_meta: Optional[dict] = {}) -> None:
        if not self.np_run is None:
            self._logger.warning("Run has already been initialized. Closing it and re-creating"
                                 " a new run.")
            self.np_run.step()
        self._logger.info("Init run")
        self.np_run = neptune.init_run(
            project=self.project,
            api_token=self.api_token,
            **init_meta
        )

    def log_dict(self, scalar_dict: dict) -> None:
        r"""Plot a dictionary of scalar-value pair."""
        # Check if Neptune run is initialized
        if self.np_run is None:
            self._logger.error("Neptune run has not been initialized. Have you run 'init_run()'?")
            return

        # Iterate over dictionary items
        for k, v in scalar_dict.items():
            if isinstance(v, (list, tuple)):
                # Log each value if it's a list or tuple
                [self.log_scalar(k, vv) for vv in v]
            else:
                # Log the scalar value
                self.log_scalar(k, v)

    def log_scalar(self, label:str, value:float):
        if isinstance(value, str):
            # ensure it's a value before making it a number
            if re.fullmatch(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$', value) is not None:
                value = float(value)

        # If value cannot be cast to a number, log it as a string
        try:
            value = float(value)
        except:
            value = str(value)

        self._logger.info(f"Logging scalar: {label}: {value}")
        self.np_run[label].append(value)

    def save_dict(self, scalar_dict: dict) -> None:
        r"""Save values instead of logging it"""
        # Check if Neptune run is initialized
        if self.np_run is None:
            self._logger.error("Neptune run has not been initialized. Have you run 'init_run()'?")
            return

        # Iterate over dictionary items
        for k, v in scalar_dict.items():
            # Log the scalar value
            self.save_value(k, v)

    def save_value(self, label: str, value: Any):
        self._logger.info(f"Logging value: {label}: {value}")
        self.np_run[label] = value

    def save_model_scalars(self, scalar_dict: Dict[str, Any]) -> None:
        r"""This saves scalars associated with model, for example, its validation performance.

        Args:
            scalar_dict (dict):
                Dictionary with str keys.

        .. note::
            For model, the scalar are assumed to be unique
        """
        if self.model is None:
            self._logger.error("Neptune model has not been initialized. Make sure the model dict was supplied "
                               "during instance creation")
            return

        for k, v in scalar_dict:
            self.model[k] = v

    def save_params(self, params: dict) -> None:
        r"""For saving the run's hyper-parameters"""
        if self.np_run is None:
            self._logger.error("Neptune run has not been initialized. Have you run 'init_run()'?")
            return

        if not isinstance(params, dict):
            raise TypeError(f"Parameters argument should be dictionary, got {type(params)} instead.")
        self.np_run['parameters'] = params

    def track_data(self, dataset_dir: Union[str, Path], version_tag):
        r"""Tracks the dataset directory to detect any changes"""
        raise NotImplementedError

    def track_model(self):

        raise NotImplementedError

    def stop(self):
        r"""Stop tracking"""
        if not self.model is None:
            self._logger.info("Stopping Neptune model tracking.")
            self.model.stop()

        if not self.np_run is None:
            self._logger.info("Stopping Neptune run.")
            self.np_run.stop()
