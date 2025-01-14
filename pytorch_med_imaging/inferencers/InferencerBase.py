import os
from abc import abstractmethod
from mnts.mnts_logger import MNTSLogger
from ..networks import *
from ..pmi_data_loader.pmi_dataloader_base import PMIDataLoaderBase
from ..solvers.SolverBase import SolverBase, SolverBaseCFG
from ..integration import TB_plotter, NP_Plotter

import torch
import torchio as tio
import ast
from dataclasses import dataclass
from typing import Union, Any, Optional
from pathlib import Path

class InferencerBase(object):
    r"""This is the base class of all inferencer, the inferencer cfg uses the same cfg as their solvers counter parts
    except they has different `required_attributes`. In addition, the inferencer also borrows some of the functions
    from :class:`SolverBase`.

    Typically, you should see the flow to be like this:

    .. mermaid::


    Attributes:
        [Required] net (torch.nn):
            The network.
        [Required] output_dir (str):
            Path to store the outputs.
        [Required] cp_load_dir (str):
            Path to the saved network state.
        use_cuda (bool, Optional):
            Whether to use GPU for computation or not. Defaul to ``True``.

    Args:
        cfg (SolverBaseCFG):
            The configuration

    See Also:
        * :class:`.SolverBase.SolverBase`
        * :class:`.SolverBase.SolverBaseCFG`

    """
    cls_cfg = SolverBaseCFG()
    def __init__(self,
                 cfg: SolverBaseCFG,
                 *args,
                 **kwargs):
        super(InferencerBase, self).__init__()

        # Defined required attributes
        self.required_attributes = [
            'output_dir',
            'batch_size',
            'net',
            'cp_load_dir',
            'data_loader'
        ]

        # borrow methods from SolverBase
        self._check_write_out_ready = SolverBase._check_fit_ready
        self._cfg = cfg

        # initialize
        self._logger        = MNTSLogger[self.__class__.__name__]
        self._load_config(cfg)   # Load config from ``cls_cfg``
        self._plotter = None

        self._logger.info("Inferencer was configured with options: {}".format(str(cfg)))

        if self.use_cuda:
            self._logger.info("Moving network to GPU.")
            self.net = self.net.cuda()

        # Flags
        self.CP_LOADED = False # whether `load_checkpoint` have been called
        self.load_checkpoint(self.cp_load_dir)



    def set_data_loader(self, data_loader: PMIDataLoaderBase):
        # SolverBase.set_data_loader(self, data_loader, None)
        self.data_loader = data_loader
        self.data_loader.run_mode = 0 # Automatically set run_mode to inference
        self.data_loader.load_dataset(exclude_augment=True)

    def _input_check(self):
        for att in self.required_attributes:
            if not hasattr(self, att):
                raise AttributeError(f"Must sepcific {str(att)} in CFG.")

    def set_plotter(self, plotter: Union[TB_plotter, str]) -> None:
        r"""Externally set :attr:`tb_plotter` manually. Note that this
        does not change

        Returns:
            TB_plotter
        """
        if not self._plotter is None:
            self._logger.warning(f"Overriding CFG ``plotter``.")
        else:
            self._plotter = None
            self.plotting = False
        self._plotter = plotter


    def load_checkpoint(self, checkpoint_path: Union[str, Path] = None) -> None:
        r"""Load the checkpoint states. If input argument is ``None``, calling this function will bring this

        Args:
            checkpoint_path (str or Path, Optional):
                Path to saved network state. If ``None``, reference attributes :attr:`cp_load_dir` instead.
        """
        if checkpoint_path is None:
            checkpoint_path = getattr(self, 'cp_load_dir', None)
            if checkpoint_path is None:
                msg = "Checkpoint load directory is not set. Configure CFG `cp_load_dir` to specify the directory to " \
                      "load the checkpoint."
                raise AttributeError(msg)

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            msg = f"Cannot open checkpoint at: {str(checkpoint_path)}!"
            if self.debug_mode:
                self._logger.warning(msg)
                self._logger.warning("Keep running because the pipeline is in debug mode.")
                self.CP_LOADED = True
                return
            else:
                raise IOError(msg)

        self._logger.info("Loading checkpoint " + str(checkpoint_path))
        self.get_net().load_state_dict(torch.load(str(checkpoint_path)), strict=False)
        self.CP_LOADED = True

    @abstractmethod
    def display_summary(self):
        raise NotImplementedError

    def write_out(self):
        r"""This is the called to perform inference. This will invoke :meth:`._write_out` with a bit of error check.
        """
        if not self.CP_LOADED:
            msg = "Checkpoint state of the network was never loaded. Have you called `load_checkpoint()`?"
            raise ArithmeticError(msg)
        else:
            self._write_out()

    @abstractmethod
    def _write_out(self):
        r"""When inheriting the inferencer class, implement this method.

        .. note::
            The method has access to attribute :attr:`output_dir` but it is not necessary a valid path. No path checking
            was implemented up until this point because different inferencer might require different output formats.
            For instance, segmentation output will generate multiple images to a same folder, whereas classification
            appends all the outputs into the same data table. Therefore, when overriding this method, be sure to perform
            path check and create the path necessary for your program to work.

        """
        raise NotImplementedError("This method must be implemented by the child class")

    def _load_config(self, cfg: SolverBaseCFG = None) -> None:
        r"""See :func:`SolverBase._load_config`."""
        SolverBase._load_config(self, cfg)
        self._input_check()
        if self.use_cuda:
            if not hasattr(self, 'batch_size_val'):
                self.batch_size_val = self.batch_size
            self.batch_size = self.batch_size_val or self.batch_size // torch.cuda.device_count()

    def _match_type_with_network(self, *args, **kwargs):
        r"""See :func:`SolverBase._match_type_with_network`."""
        return SolverBase._match_type_with_network(self, *args, **kwargs)

    def _unpack_minibatch(self, *args, **kwargs):
        r"""See :func:`SolverBase._unpack_minibatch`."""
        return SolverBase._unpack_minibatch(self, *args, **kwargs)

    def _check_write_out_ready(self):
        r"""See :func:`SolverBase._check_write_out_ready`."""
        return SolverBase._check_fit_ready(self)

    def get_net(self):
        r"""See :func:`SolverBase.get_net<pytorch_med_imaging.solvers.SolverBase.get_net>`."""
        return SolverBase.get_net(self)

    def _prepare_data(self):
        r"""Try to load in training mode first to include the ground-truth but ignoring the augmentation."""
        try:
            self._inference_subjects = self.data_loader._load_data_set_training(True)
        except:
            self._inference_subjects = self.data_loader._load_data_set_inference()