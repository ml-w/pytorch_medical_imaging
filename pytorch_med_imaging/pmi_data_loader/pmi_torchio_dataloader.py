import os
import torchio as tio
from .pmi_dataloader_base import PMIDataLoaderBase, PMIDataLoaderBaseCFG
from .pmi_image_dataloader import PMIImageDataLoader, PMIImageDataLoaderCFG
from .. import pmi_data
from ..pmi_data import ImageDataSet, PMIDataBase
from .lambda_tio_adaptor import CallbackQueue
from .computations.queue_callback import *
from typing import *
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from collections.abc import Iterable
import re
import pandas as pd


__all__ = ['PMITorchioDataLoaderCFG', 'PMITorchioDataLoader']

class PMITorchioDataLoaderCFG(PMIImageDataLoaderCFG):
    r"""Configuration for :class:`PMIImageDataLoader`.

    Class Attributes:
        input_data (dict):
            Input data. Must be either PMI data or iterable.
        input_dtypes (dict):
            Type of input data.
        sampler (str, Optional):
            Determine the ``tio.Sampler`` used to sample the images. Support ['weighted'|'uniform'|'grid'] currently.
            Default to ``None``, which means no sampler is used (i.e., the whole image is loaded).
        sampler_kwargs (dict, Optional):
            The kwargs passed to ``tio.Sampler``. For 'weighted', key ``patch_size`` and ``prob_map`` is required. For
            'uniform', only ``patch_size`` is required. Unless sampler is ``None``, this needs to be specified. Default
            value is only a place holder.
        augmentation (str, Optional):
            Path to yaml file to create the ``tio.Compose`` transform. Default to ``None``.
        patch_sampling_callback (Callable or str, Optional):
            A function that is called after patch sampling to generate new data from sampled patches. For example, if
            texture features is required after the patches are sampled, you can assign the function to compute the
            texture features using this setting. This should be used with ``patch_sampling_callback_kwargs`` and also
            ``create_new_attribute``. Default to ``None``.
        patch_sampling_callback_kwargs (dict, Optional):
            The kwargs that will be supplied to the lambda function specified by ``patch_sampling_callback``.
            Default to empty dict.
        create_new_attribute (str, Optional):
            The new data created by `patch_sampling_callback` will be attached to the subject using this argument
            as the attribute name. The new data can then be accessed by ``tio.Subject()[create_new_attribute]``.
            Default to ``None``.
        inf_samples_per_vol (int, Optional):
            Sometimes inference will require more sampled patches to generate appealing results (e.g, segmentation),
            this argument,

    .. hint::
        To use `weighted` sampler, you must submit LabelMap input as `probmap` key for :attr:`input_data` and
        `input_dtypes`.

    .. warning::
        Note that this class does not use the same logic as the rest and does not respsect `input_dir`, `target_dir` and
        `output_dir`. There woun't be any effects if you specify them.


    See Also:
        * :class:`PMIDataLoaderBaseCFG`

    Examples:
        1. Directly declare the data:

            >>> data_label = DataLabel(...)
            >>> img = ImageDataSet(...)
            >>> data_loader_cfg = PMITorchioDataLoaderCFG(
            >>>     input_data = {
            >>>         'labels': data_label,
            >>>         'image': img}
            >>> )

        2. If path is given, they are considered as imaging data, but you also need `input_dtypes` to specify the
           data type (ScalarImage or LabelMap)

           >>> data_loader_cfg = PMI
           >>> PMITorchioDataLoaderCFG(
           >>>      input_data = {
           >>>         'labels': data_label,
           >>>         'image': img
           >>>      },
           >>>      input_dtypes = {
           >>>          'labels': 'uint8'
           >>>      }
           >>> )
           >>>
           >>>

    """
    input_data: Dict[str, Union[PMIDataBase, Iterable[Any]]] = None
    input_dtypes: Dict[str, Union[str, Type]] = {}


class PMITorchioDataLoader(PMIImageDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_input(self):
        super()._check_input()
        if set(self.input_dtypes.keys()) in set(self.input_data.keys()):
            raise KeyError("Keys of input dtypes are not in input.")

    def _prepare_data(self) -> dict:
        # Load data if the input data are Path
        data = {}
        ids = {}
        for k, v in self.input_data.items():
            if isinstance(v, PMIDataBase):
                data[k] = v
                continue
            elif isinstance(v, (str, Path)):
                v = Path(v)
                if not v.is_dir():
                    raise TypeError("String is specified as input but does not lead to image directory.")
                # If directory, treat it as image path
                _type = self.input_dtypes.get(k, 'float')
                _data = self._read_image(v, dtype=_type)
                data[k] = _data
            elif isinstance(v, Iterable):
                self._logger.warning("Input iterable data detected. Note that automatic ordering for "
                                     "custom iterable data is not supported.")
                data[k] = _v

        # Make sure the ID list are correctly configured
        for k, v in data.items():
            if isinstance(v, PMIDataBase):
                ids[k] = v.get_unique_IDs()

        # Create a DataFrame from the IDs
        ids_df = pd.concat([pd.Series([1] * len(v), index=v, name=k)for k, v in ids.items()],
                           join='outer', axis=1)
        if ids_df.isna().any().any():
            self._logger.warning("IDs are not properly aligned")
            self._logger.warning('\n' + ids_df.fillna("Missing").to_string())
        # Note: No need to check length because ids right = length right
        return data

    def _configure_sampler(self):
        r"""Weighted sampler needs probmap to establish sampling region,"""
        if self.sampler == 'weighted':
            if not 'probmap' in self.input_data:
                msg = "Key `probmap` must be populated with LabelMap"
                self._logger.error(msg)
                raise KeyError(msg)
            else:
                # Place holder
                self.probmap_dir = 1
        super()._configure_sampler()
        
