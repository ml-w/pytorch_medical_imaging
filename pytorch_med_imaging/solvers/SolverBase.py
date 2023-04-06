import ast
import re
import os
import time
import pprint
from abc import abstractmethod

import gc
import numpy as np
import torch
from torch.optim import lr_scheduler
from typing import Union, Iterable, Optional
from pathlib import Path

import torchio as tio
import torch.distributed as dist
from tqdm import tqdm

from ..pmi_base_cfg import PMIBaseCFG
from ..tb_plotter import TB_plotter
from .. import lr_scheduler as pmi_lr_scheduler
from ..lr_scheduler import PMILRScheduler
from ..pmi_data_loader import PMIDataLoaderBase, PMIDistributedDataWrapper
from .earlystop import BaseEarlyStop
from ..loss import *


__all__ = ['SolverBaseCFG', 'SolverBase']

class SolverBaseCFG(PMIBaseCFG):
    r"""Configuration for initializing :class:`SolverBase` and its child classes.

    Class Attributes:
        [Required] net (torch.nn.Module):
            The network.
        [Required] data_loader (pmi_dataloader):
            Dataloader for training iteration
        [Required] optimizer (nn.Module or str):
            Optimizer for training. A string can be used to create the optimizer. See :func:`create_optimizer`.
        [Required] loss_function (nn.Module):
            Loss function for training. Note that some child class might specify a default value in their
            CFG, but if it doesn't, this should always be specified.
        [Required] init_lr (float):
            Initial learning weight.
        [Required] batch_size (int):
            Mini-batch size.
        [Required] num_of_epochs (int):
            Number of epochs.
        [Required] unpack_key_forowrad (list of str):
            This list is used to unpack the ``tio.Subject`` (or patches) loaded by the ``data_loader``. See
            :func:`.solve_epoch`.
        batch_size_val (int, Optional):
            If validation set is loaded, this parameters is used as the batch size for the validation loop. If this is
            not specified, the default is to use `self.batch_size` instead.
        cp_save_dir (str, Optional):
            Specify the file directory to write the network parameters when criteria reached. If ``None``, you will
            need to call the :func:`.fit` with the `checkpoint_path` argument. Default to ``None``.
        cp_load_dir (str, Optional):
            Specify the direcotry where the checkpoint is to be loaded. If not ``None``, the checkpoint is loaded before
            calling :func:`.fit`.
        output_dir (str, Optional):
            Required for :class:`.InferencerBase` to decide where the outputs are written to. Default to ``None``.
        unpack_key_inference (Iterable[str], Optional):
            If this is specified, the inferencer will be default to use this, otherwise, it will try to use
            :attr:`unpack_key_forward`, but this is not always correct. Default to ``None``.
        compile_net (bool, Optional):
            If `True`, the network will be compiled using `torch.compile`. Support only for torch version >= 2.0
        init_mom (float, Optional):
            Initial momentum. Ignored for some of the optimizers.
        lr_sche (lr_scheduler._LRScheduler, Optional):
            Dictates how the LR is changed during the course of training. Stepped after each epoch is done. Default
            to ``lr_scheduler.ExponentialLR`` supplied by ``torch``. Some additional schedulers is also implemented
            in the ``pmi_lr_scheduler`` module.
        lr_sche_args (list, Optional):
            Some schedulers require supplying arguments. Do it through this attribute. Default to empty list ``[]``.
        lr_sche_kwargs (dict, Optional):
            Some schedulers require supplying key word arguments. Do it through this attribute. Defaul to ``{}``.
        use_cuda (bool, Optional):
            Whether this solver will move the items to cuda for computation. Default to ``True``.
        debug (bool, Optional):
            Whether the current session is executed with debug mode. Passed to ``data_loader`` mainly. Default to
            ``False``.
        dataloader_val (PMIDataLoaderBase, Optional):
            The dataloader used during validation. If it is not provide, the validation step will be skipped and the
            training loss will be used to identify early stopping time points. Default to ``None``.
        early_stop (BaseEarlyStop, Optional):
            If not ``None``, this will specify the policy for early stopping. See :class:`:class:.earlystop.BaseEarlyStop`
        early_stop_args(list, Optional):
            Pass to initializing :class:`.earlystop.BaseEarlyStop`. Defaul to ``[]``.
        early_stop_kwargs(dict, Optional):
            Pass to initializing: class:`.earlystop.BaseEarlyStop` as keyword arguments. Default to ``{}``.
        accumulate_grad (int, Optional):
            If value > 1 specified, gradient will be calculate with loss accumulated for multiple iterations. See
            :func:`SolverBase._update_network`.
        plotter_dict (dict, Optional):
            This dict could be used by the child class to perform plotting after validation or in each step.
        plot_to_tb (bool, Optional):
            If try, the solver will try to create a :class:`TB_plotter` for plotting the intermediate results.

    """
    # Training hyper params (must be provided for training)
    init_lr       : float = None
    num_of_epochs : int   = None
    batch_size    : int   = None
    batch_size_val: int   = None

    # I/O
    unpack_key_forward  : Iterable[str]           = None
    unpack_key_inference: Optional[Iterable[str]] = None
    cp_save_dir         : Optional[str]           = None
    cp_load_dir         : Optional[str]           = None
    output_dir          : Optional[str]           = None

    net          : torch.nn.Module       = None
    loss_function: torch.nn              = None
    optimizer    : torch.optim.Optimizer = None
    data_loader  : PMIDataLoaderBase     = None

    # Options with defaults
    use_cuda         : Optional[bool]              = True
    debug_mode       : Optional[bool]              = False
    compile_net      : Optional[bool]              = False
    data_loader_val  : Optional[PMIDataLoaderBase] = None
    lr_sche          : Optional[PMILRScheduler]    = None # If ``None``, lr_scheduler.ExponentialLR will be used.
    lr_sche_args     : Optional[list]              = []
    lr_sche_kwargs   : Optional[dict]              = {}
    plotter_dict     : Optional[dict]              = None
    early_stop       : Optional[BaseEarlyStop]     = None
    early_stop_args  : Optional[list]              = None
    early_stop_kwargs: Optional[dict]              = None
    accumulate_grad  : Optional[int]               = 1
    init_mom         : Optional[float]             = None
    plot_to_tb       : Optional[bool]              = False

    def __str__(self):
        _d = {k: v for k, v in self.__dict__.items() if k[0] != '_'}
        _d['net'] = self.net._get_name()
        return pprint.pformat(_d, indent=2)



    @property
    def solver_cls(self) -> type:
        return ast.literal_eval(self.__class__.__name__.replace('CFG', ''))

    @property
    def inferencer_cls(self) -> type:
        return ast.literal_eval(self.__class__.__name__.replace('SolverCFG', 'Inferencer'))


class SolverBase(object):
    """Base class for all solvers. This class must be inherited before it can work properly. The child
    classes should inherit the abstract methods. The initialization is done by specifying class attributes of
    :class:`SolverBaseCFG`.

    Method call sequence
    ^^^^^^^^^^^^^^^^^^^^

    .. mermaid::

        sequenceDiagram
            autonumber
            actor user
            participant fit()
            participant solve_epoch()
            participant step()
            participant validation()
            user ->>+ fit(): call
            fit() ->>+ solve_epoch(): call
            solve_epoch() ->> solve_epoch(): _epoch_prehook()
            solve_epoch() ->>+ step(): call
            step() ->> step(): Process one mini-batch
            step() ->>- solve_epoch(): Return loss
            solve_epoch() ->> solve_epoch(): _step_callback()
            opt has data_loader_val
                solve_epoch() ->>+ validation(): call
                Note over validation(): After each validation mini-batch
                validation() ->> validation(): _validation_step_callback()
                Note over validation(): After validate who set of data
                validation() ->> validation(): _validation_callback()
                validation() ->>- solve_epoch(): Return validation loss
            end
            solve_epoch() ->> solve_epoch(): _epoch_callback()
            Note right of solve_epoch(): Plot to tensorboard
            Note right of solve_epoch(): Step early stopper if exist
            solve_epoch() ->>- fit(): Return train/validation loss
            opt if loss < last_min_loss
                fit() ->> fit(): Save network state
                Note right of fit(): Update `last_min_loss`
            end
            fit() ->>- user: Finish

    Class Attributes:
        cls_cfg (SolverBaseCFG):
            The config parameters.

    Attributes:
        _step_called_time (int):
            Number of time :func:`step` was called.
        _decayed_time (int):
            Number of time :func:`decay_optimizer` was called.
        _tb_plotter (TB_plotter):
            Tensor board summary writter.

    .. hint::
        When implementing a child class solver, you should pay attention to implementing the two methods including
        :func:`._validation_step_callback` and :func:`._validation_callback`. The step callback allows you to store the
        results of each step in the validation, the callback allows you to compute the performance of this epoch. Using
        these results, you can decide in the :func:`.solve_epoch` the criteria for saving the state of the network.

    """
    cls_cfg = SolverBaseCFG
    def __init__(self, cfg: SolverBaseCFG,
                 *args, **kwargs):
        super(SolverBase, self).__init__()
        self._logger        = MNTSLogger[self.__class__.__name__]
        self._load_config(cfg)   # Load config from ``cls_cfg``

        # Define minimal requirement to kick start a ``fit()``
        self._required_attributes = {
            'net': torch.nn.Module,
            'data_loader': (PMIDataLoaderBase, torch.utils.data.DataLoader, PMIDistributedDataWrapper),
            'optimizer': torch.optim.Optimizer,
            'loss_function': torch.nn.Module,
            'init_lr': float,
            'num_of_epochs': int,
            'unpack_key_forward': (tuple, list),
            'batch_size': int
        }

        # Optimizer attributes
        self._step_called_time: int = 0
        self._decayed_time    : int = 0
        self._accumulated_steps: int = 0

        # internal attributes
        self._tb_plotter = None
        self.current_epoch = 0

        # external_att
        self.plotter_dict      = {}

         # create loss function if not specified
        self.prepare_lossfunction()
        self.create_optimizer()


        self._logger.info("Solver was configured with options: {}".format(str(cfg)))
        if  len(kwargs):
            self._logger.warning("Some solver configs were not used: {}".format(kwargs))

        if self.cp_load_dir is not None:
            self.load_checkpoint(self.cp_load_dir)

        if self.use_cuda:
            self._logger.info("Moving lossfunction and network to GPU.")
            if not dist.is_initialized():
                self.loss_function = self.loss_function.cuda()
                self.net = self.net.cuda()
            else:
                # had to use this way to avoid running out of CUDA mem
                self.loss_function = self.loss_function.cuda(device=dist.get_rank())
                self.net = self.net.cuda(device=dist.get_rank())

        if self.plot_to_tb:
            self.create_plotter()

        # Stopping criteria
        if isinstance(self.early_stop, str):
            self.set_early_stop(self.early_stop)

    def _load_config(self, config_file = None):
        r"""Function to load the configurations. If ``config_file`` is ``None``, load the default class
        :class:`SolverBaseCFG` that is stored as a class attribute :attr:`cls_cfg`.
        """
        # Loading basic inputs
        if not config_file is None:
            # cls_dict = { attr: getattr(cls, attr) for attr in dir(cls)}
            self.__dict__.update(config_file.__dict__)
            self.__class__.cls_cfg = config_file
        else:
            self._logger.warning("_load_config called without arguments.")

    def _check_fit_ready(self) -> bool:
        r"""Check the instance attribute specified in ``self._required_attribute`` and their types to make sure that the
        attributes required in :func:`fit` have all been specified correctly.

        Returns:
             bool: ``True`` if attributes is readied for :func:`fit` to run. An exception is raised otherwise.
        """
        for att_name, att_type in self._required_attributes.items():
            if not isinstance(getattr(self, att_name), att_type):
                msg = f"Expect '{att_name}` to be type {att_type} but got {getattr(self, att_name)}."
                raise TypeError(msg)
                return False
        return True

    def get_net(self) -> torch.nn.Module:
        r"""Return ``self.net`` or ``self.net.module`` if parallel network is invoked.

        Returns:
            torch.nn.module
        """
        if torch.cuda.device_count() > 1:
            try:
                return self.net.module
            except AttributeError:
                return self.net
        else:
            return self.net

    def get_optimizer(self) -> torch.optim.Optimizer:
        r"""Return ``self.optimizer``.

        Returns:
            torch.nn.optim.Optimizer
        """
        return self.optimizer

    def set_data_loader(self,
                        data_loader: PMIDataLoaderBase,
                        data_loader_val: PMIDataLoaderBase = None) -> None:
        r"""Externally set the dataloaders if they were not specified in ``self.cls_cfg``.

        Args:
            data_loader (PMIDataLoaderBase):
                Replaces ``self.data_loader``.
            dataloader_val (PMIDataLoaderBase):
                Replaces ``self.data_loader_val``
        """
        # If self.data_loader was never configured, make it `None`.
        if not hasattr(self, 'data_loader'):
            self.data_loader = None

        # If self.data_loader is not None, its an overriding event.
        if not self.data_loader is None:
            self._logger.warning("Overriding CFG `dataloader`.")

        # Check input type
        if not isinstance(data_loader, (PMIDataLoaderBase, PMIDistributedDataWrapper)):
            raise TypeError(f"Expect input to be ``PMIDataLoaderBase`` for ``data_loader``, "
                            f"but got: {type(data_loader)}")
        # If solver is in DDP mode, postpone the loading
        if not dist.is_initialized():
            self.data_loader = data_loader.get_torch_data_loader(self.batch_size)
        else:
            self.data_loader = data_loader

        # Do the same for data_loader_val
        if not hasattr(self, 'data_loader_val'):
            self.data_loader_val = None

        if not self.data_loader_val is None and not data_loader_val is None:
            self._logger.warning("Overriding CFG `dataloader_val`.")

        if not data_loader_val is None:
            if not isinstance(data_loader_val, PMIDataLoaderBase) and not data_loader_val is None:
                raise TypeError(f"Expect input to be ``PMIDataLoaderBase`` for ``data_loader_val``, "
                                f"but got: {type(data_loader_val)}")
            # In DDP only rank 0 needs to load val
            if dist.is_initialized():
                if dist.get_rank() != 0:
                    return
            self.data_loader_val = data_loader_val.get_torch_data_loader(self.batch_size_val or self.batch_size,
                                                                         exclude_augment=True)

    def set_lr_scheduler(self,
                         scheduler: Union[str, PMILRScheduler],
                         *args, **kwargs) -> None:
        r"""Externally set the :attr:`lr_sche` manually.

        Args:
            scheduler (str or PMILRScheduler):
                If a ``str`` is supplied, this will try to create the learning rate scheduler using the
                :class:`PMILRScheduler` and store it as :attr:`lr_sche`.
            *args:
                Pass to PMILRScheduler if ``scheduler`` is a string.
            **kwargs:
                Pass to PMILRScheduler if ``scheduler`` is a string.


        See Also:
            * :class:`PMILRScheduler`
        """
        if not self.lr_sche is None and not isinstance(self.lr_sche, str):
            self._logger.warning("Overriding CFG ``lr_sche``.")

        # if a string is supplied, try to creat it in PMILRScheduler.
        if isinstance(scheduler, str):
            if len(args) == 0:
                args = self.lr_sche_args
            if len(kwargs) == 0:
                kwargs = self.lr_sche_kwargs
            self.lr_sche = pmi_lr_scheduler.PMILRScheduler(scheduler, *args, **kwargs)
        else:
            self.lr_sche = scheduler
        self.lr_sche.set_optimizer(self.optimizer)

    def set_early_stop(self,
                       early_stop: Union[str, BaseEarlyStop],
                       *args,
                       **kwargs) -> None:
        r"""Externally set the :attr:`early_stop`. Also used in __init__ to create the instance if the cfg
        ``early_stop`` attribute is a string. See :class:`.early_stop_scheduler.BaseEarlyStop` for more.

        Args:
            early_stop (str or BaseEarlyStop):
                The early stop instance.
            *args:
                The argument that will be passed to :class:`BaseEarlyStop`.
            **kwargs:
                The key word arguments that will be passed to :class:`BaseEarlyStop`.

        """
        if not self.early_stop is None and not isinstance(self.early_stop, str):
            self._logger.warning("Overriding CFG ``early_stop``.")

        # if a string is supplied, try to creat it in PMILRScheduler.
        if isinstance(early_stop, str):
            if len(args) == 0:
                args = self.early_stop_args or []
            if len(kwargs) == 0:
                kwargs = self.early_stop_kwargs or {}
            self.early_stop = BaseEarlyStop.create_early_stop_scheduler(self.early_stop, *args, **kwargs)
        else:
            self.early_stop = early_stop

    def set_plotter(self, plotter: Union[TB_plotter, str]) -> None:
        r"""Externally set :attr:`tb_plotter` manually.

        Returns:
            TB_plotter
        """
        if not self._tb_plotter is None:
            self._logger.warning(f"Overriding CFG ``tb_plotter``.")
        self._tb_plotter = plotter

    def create_plotter(self):
        r"""Create the tensorboard plotter."""
        try:
            # for legacy purpose, this has always been specified by global env variable.
            tensorboard_rootdir =  Path(os.environ.get('TENSORBOARD_LOGDIR', '/media/storage/PytorchRuns'))
            if not tensorboard_rootdir.is_dir():
                self._logger.warning("Cannot read from TENORBOARD_LOGDIR, retreating to default path...")
                tensorboard_rootdir = Path("/media/storage/PytorchRuns")

            # Strip the parenthesis and comma from the net name to avoid conflicts with system
            net_name = str(self.net)
            net_name = net_name.translate(str.maketrans('(),','[]-'," "))
            self._logger.info("Creating TB writer, writing to directory: {}".format(tensorboard_rootdir))

            # check if the target exist
            dirs_num = [int(re.search(f"{net_name}-(?P<number>\d+)", d.name).groupdict()['number'])
                        for d in tensorboard_rootdir.glob(f"{net_name}-*")]
            if len(dirs_num) > 0:
                next_num = max(dir_num) + 1
            else:
                next_num = 0

            writer = SummaryWriter(str(tensorboard_rootdir.joinpath(
                "{}{}".format(net_name,
                              f'-{next_num}' if next_num > 0 else ''
            ))))

            self._tb_plotter = TB_plotter(writer)
        except Exception as e:
            self._logger.warning("Tensorboard writter creation encounters failure, falling back to no writer.")
            self._logger.exception(e)
            self._tb_plotter = None

    def net_to_parallel(self) -> None:
        r"""Call to move :attr:`net` to :class:`torch.nn.DataParallel`. Sometimes the network used is special and
        you can override this for those exceptions.
        """
        if dist.is_initialized():
            raise ArithmeticError("DDP should not use this method to distribute the network.")
        if (torch.cuda.device_count()  > 1) & self.use_cuda:
            self._logger.info("Multi-GPU detected, using nn.DataParallel for distributing workload.")
            self.net = nn.DataParallel(self.net)
            if self.compile_net:
                if int(torch.__version__.split('.')[0]) < 2:
                    self._logger.warning("Compile net only supported for torch version >= 2.0.0.")
                else:
                    self.net = torch.compile(self.net)

            self.create_optimizer(self.net)

    def set_loss_function(self, func: torch.nn.Module) -> None:
        r"""Externally set :attr:`loss_function` manually. This will also move the loss function to GPU if
        :attr:`use_cuda` is ``True``.

        Args:
            func (torch.nn.Module):
                The loss function.

        """
        self._logger.debug("loss functioning override.")
        if self.use_cuda:
            try:
                if dist.is_initialized():
                    func = func.cuda(device=dist.get_rank())
                else:
                    func = func.cuda()
            except:
                self._logger.warning("Failed to move loss function to GPU")
                pass
        del self.loss_function
        self.loss_function = func

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        r"""Load the stored parameters for :attr:`net`.

        Args:
            checkpoint_dir (str):
                Directory to the stored parameters file. This file should be created using the default pytorch method
                :func:`torch.save`.

        """
        if os.path.isfile(checkpoint_dir):
            # assert os.path.isfile(checkpoint_load)
            try:
                self._logger.info("Loading checkpoint " + checkpoint_dir)
                self.get_net().load_state_dict(torch.load(checkpoint_dir,
                                                          torch.device('cpu')), strict=False)
            except Exception as e:
                if not self.debug_mode:
                    self._logger.error(f"Cannot load checkpoint from: {checkpoint_dir}")
                    raise e
                else:
                    self._logger.warning(f"Cannot load checkpoint from {checkpoint_dir}")
        else:
            self._logger.warning("Checkpoint specified but doesn't exist!")
            self._logger.debug(f"{checkpoint_dir}")

    def prepare_lossfunction(self) -> None:
        r"""This method is the default way of preparing the loss function, basically sets :attr:`class_weights` into
        :attr:`loss_function.weight`. This allow us to port the class weights as fine-tunable hyperparameters.
        """
        if self.loss_function is None:
            raise AttributeError("Loss function must be defined!")

        if self.class_weights is None and self.loss_function.weight is None:
            try:
                self.auto_compute_class_weights()
            except Exception as e:
                msg = "No class weights specified and automatic weight computation failed. It is strongly recommend to " \
                      "use class weights for training."
                self._logger.warning(msg)
                if self._logger.log_level == 10:
                    self._logger.error("Original error raised:")
                    self._logger.exception(e)
        elif self.class_weights is not None and self.loss_function.weight is not None:
            # warn if weight is being override
            msg = f"Overwriting class weights using CFG inputs: {self.loss_function.weight} -> {self.class_weights}"
            self._logger.warning(msg)
            self.loss_function.weight.copy_(torch.as_tensor(self.class_weights).float())
        elif self.class_weights is not None and self.loss_function.weight is None:
            self.loss_function.weight = torch.as_tensor(self.class_weights).float()

        # make sure the weights are float
        if not self.loss_function.weight is None:
            self.loss_function.weight = self.loss_function.weight.float()

    def step(self, *args):
        r"""This function executes one step in a training loop, which includes:

        1. forward run
        2. loss computation
        3. backwards propagation and updating the network

        Customized solver can inherit this to alter the behavior of each training step with ease.

        Args:
            *args:
                Argument passed to :func:`_feed_forward`.

        .. note::
            If the learning rate scheduler (:attr:`lr_sche`) is :class:`lr_scheduler.OneCycleLR`, this method will also
            invoke steping the learning rate scheduler. Otherwise, the learning rate scheduler is only stepped after
            each epoch.

        .. hint::
            Override this function in the child class to customize the training behavior.

        """
        out = self._feed_forward(*args)
        loss = self._loss_eval(out, *args)

        self._update_network(loss)

        # if schedular is OneCycleLR
        if isinstance(self.lr_sche, lr_scheduler.OneCycleLR):
            self.lr_sche.step()
        self._step_called_time += 1
        return out, loss.cpu().data

    def _update_network(self, loss: torch.Tensor):
        r"""Perform back propagation and then updates the network parameters.

        Args:
            loss (torch.Tensor):
                This should be a single-valued :class:`torch.Tensor` computed by the loss function. This should have
                gradient computation hooks.

        .. note::
            If :attr:`accumulate_grad` > 1, gradient accumulation will be used, i.e., the network will be upgraded based
            on the averaged loss over :attr:`accumulated_grad` run of :func:`step`.

        """
        # Gradient accumulation
        if self.accumulate_grad > 1:
            self._accumulated_steps += 1
            loss = loss / float(self.accumulate_grad)
            loss.backward()
            # loss.detach_()
            if self._accumulated_steps >= self.accumulate_grad:
                self._logger.debug("Updating network params from accumulated loss.")
                self.optimizer.step()
                self.optimizer.zero_grad()
                self._accumulated_steps = 0
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def create_optimizer(self,
                         net: torch.nn.Module = None,
                         optimizer: Union[str, torch.optim.Optimizer] = None) -> torch.optim.Optimizer:
        r"""Create the optimizer from string or set the network params into the optimizer if a class is provided.

        Args:
            net (torch.nn.Module, Optional):
                If not ``None``, replace that specified in the CFG. Default to ``None``.
            optimizer (str or torch.optim.Optimizer or class):
                If not ``None``, this method will replace the optimizer set by using the CFG class. If a string is
                provided, the value should be one of ['Adam'|'SGD'|'AdamW'], which will create the optimizer with
                respect to :attr:`init_lr` and :attr:`init_mom` depending on the type of optimizer. If a
                ``torch.optim.Optimizer`` instance is provided in the CFG, do nothing and assume the network parameters
                have already been set to the optimizer. Default to ``None``.

        Returns:
            torch.optim.Optimizer
        """
        if not net is None and net != self.net:
            msg = f"Overriding network when creating optimizer. If you didn't inherit the function create_optimizer, " \
                  f"something might have gone wrong becuase network should have already been created!"
            self._logger.warning(msg)
            self.net = net

        if not optimizer is None and not self.optimizer is None and not isinstance(self.optimizer) is str:
            msg = f"Overriding optimizer defined in CFG. If you didn't inherit the function create_optimizer, something" \
                  f"might have gone wrong!"
            self._logger.warning(msg)
            self.optimizer = optimizer

        if isinstance(self.optimizer, str):
            net_params = self.net.parameters()
            if self.optimizer == 'Adam':
                self.optimizer = torch.optim.Adam(net_params, lr=self.init_lr)
            elif self.optimizer == 'AdamW':
                self.optimizer = torch.optim.AdamW(net_params, lr=self.init_lr)
            elif self.optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(net_params, lr=self.init_lr,
                                             momentum=self.init_mom)
            else:
                raise AttributeError(f"Expecting optimzer to be one of ['Adam'|'SGD'|'AdamW']")
        elif not isinstance(self.optimizer, torch.optim.Optimizer):
            msg = f"Expect optimizer to be either a string or a torch optimizer, but got {type(self.optimizer)} " \
                  f"instead. Check your settings in the CFG class"
            raise TypeError(msg)
        else:
            # Do nothing if everything is fine. Assume the optimizer already knows the network parameters.
            pass

        if isinstance(self.lr_sche, str):
            self.set_lr_scheduler(self.lr_sche)
        elif self.lr_sche is None:
            self.set_lr_scheduler('ExponentialLR', 0.99)
        elif isinstance(self.lr_sche, PMILRScheduler):
            self.lr_sche.set_optimizer(self.optimizer)

        return self.optimizer

    def decay_optimizer(self, *args):
        r"""Step learning rate after the optimizer has been stepped.

        Args:
            *args:
                All arguments are passed to :func:`lr_scheduler.step`

        .. note::
            Some of the learning rate schedulers require different ``step()`` inputs. It is currently hard-coded in this
            function which might cause some trouble. Only :class:`lr_scheduler.ReduceLROnPlateau` and
            :class:`lr_scheduler.OneCycleLR` is implemented.
        """
        if not self.lr_sche is None:
            if isinstance(self.lr_sche, (lr_scheduler.ReduceLROnPlateau)):
                self.lr_sche.step(*args)
            elif isinstance(self.lr_sche, lr_scheduler.OneCycleLR):
                # Do nothing because it's supposed to be done in `step()`
                pass
            else:
                self.lr_sche.step()
        self._decayed_time += 1

        # ReduceLROnPlateau has no get_last_lr attribute, thus some exceptions was written in get_last_lr() to work
        # arround this problem.
        lass_lr = self.get_last_lr()
        self._logger.debug(f"Decayed optimizer, new LR: {lass_lr}")

    def get_last_lr(self) -> float:
        r"""Return the last training loss/validation loss processed by the optimizer.

        Returns:
            float
        """
        try:
            lass_lr = self.lr_sche.get_last_lr()[0]
        except TypeError:
            lass_lr = self.lr_sche.get_last_lr()
        except AttributeError:
            if isinstance(self.get_optimizer().param_groups, (tuple, list)):
                lass_lr = self.get_optimizer().param_groups[0]['lr']
            else:
                lass_lr = next(self.get_optimizer().param_groups)['lr']
        except Exception as e:
            self._logger.warning("Cannot get learning rate!")
            self._logger.exception(e)
            lass_lr = "Error"
        return lass_lr

    def get_last_train_loss(self) -> Union[float, None]:
        r"""Return the loss from last epoch. If it hasn't been computed, return ``None`` instead.

        Returns:
            float
        """
        try:
            return self._last_epoch_loss
        except AttributeError:
            return None

    def get_last_val_loss(self) -> Union[float, None]:
        r"""Return the loss from last validation loop. If it hasn't been computed, return ``None`` instead.

        Returns:
            float
        """
        try:
            return self._last_val_loss
        except AttributeError:
            return None

    def test_inference(self, *args):
        r"""This function is for testing only. """
        with torch.no_grad():
            out = self.net.forward(*list(args))
        return out

    def solve_epoch(self, epoch_number):
        r""" Run one epoch, i.e., perform training iterating the ``data_loader`` for whole set of data.

        Args:
            epoch_number (int):
                The number of the current epoch for calculation of various training policies e.g., LR decay.

        .. hint::
            :func:`_epoch_callback` will be executed after the epoch iteration ends, right before running
            :func:`decay_optimizer`. In this base class, the callback only plots the results of the epoch to
            tensorboard, but it has other potentials that can be exploited when implementing the child classes.

        .. caution::
            If you inherit this, remember to align the definition of :attr:`_last_val_loss` and
            :attr:`_lass_epoch_loss`. Otherwise, checkpoint saving and early stopping schedulers migth break.

        """
        self.current_epoch = epoch_number
        self._epoch_prehook()
        E = []
        # Reset dict each epoch
        self.net.train()
        self.optimizer.zero_grad() # make sure validation loop doesn't alter the gradients
        self.plotter_dict = {'scalars': {}, 'epoch_num': epoch_number}

        if not dist.is_initialized():
            data_loader = self.data_loader.get_torch_data_loader(self.batch_size) \
                if isinstance(self.data_loader, PMIDataLoaderBase) else self.data_loader
        else:
            data_loader = self.data_loader.get_torch_data_loader(self.batch_size)
        for step_idx, mb in enumerate(data_loader):
            s, g = self._unpack_minibatch(mb, self.unpack_key_forward)

            # initiate one train step. Things should be plotted in decorator of step if needed.
            out, loss = self.step(s, g)
            E.append(loss.data.cpu())
            self._logger.info("\t[Step %04d] loss: %.010f"%(step_idx, loss.data))

            self._step_callback(s.cpu(), g.cpu(), out.detach().cpu().float(), loss.data.cpu(),
                                step_idx=epoch_number * len(data_loader) + step_idx,
                                uid = mb.get('uid', None))
            del s, g, out, loss, mb
            gc.collect()
        epoch_loss = np.array(E).mean()
        self.plotter_dict['scalars']['Loss/Loss'] = epoch_loss
        self._last_epoch_loss = epoch_loss

        self._logger.info("Initiating validation.")
        self.validation()
        self._epoch_callback()
        self.decay_optimizer(epoch_loss)

    def validation(self) -> list:
        r"""Default pipeline for running the validation. This introduce two class attribute lists
        :attr:`validation_losses` and :attr:`perfs`. They are to be used in :func:`_validation_step_callback` and also
        :func:`_validation_callback`.

        .. hint::
            You can inherit :func:`_validation_step_callback` and :func:`_validation_callback` to compute the step
            performance and validation performance. The key is that loss has to be stored during the step callback for
            calculation of the validation average loss.

        See Also:
            * :func:`_validation_step_callback`
            * :func:`_validation_callback
        """
        if self.data_loader_val is None:
            self._logger.warning("Validation skipped because no loader is available.")
            return None

        with torch.no_grad():
            self.validation_losses = []
            self.perfs = []
            self.get_net().eval()
            for mb in tqdm(self.data_loader_val, desc="Validation", position=2):
                s, g = self._unpack_minibatch(mb, self.unpack_key_forward)
                s = self._match_type_with_network(s)
                g = self._match_type_with_network(g) # no assumption but should be long in segmentation only.

                if isinstance(s, list):
                    res = self.get_net().forward(*s)
                else:
                    res = self.get_net().forward(s)

                loss = self._loss_eval(res, s, g.squeeze().long())
                self._logger.debug(f"_val IDs: {mb['uid']}") if not mb.get('uid', None) is None else None
                self._logger.debug("_val_step_loss: {}".format(loss.cpu().data.item()))

                uids = mb.get('uid', None)
                self._validation_step_callback(g.cpu(), res.cpu(), loss.cpu(), uids)
                del mb, s, g, loss
                gc.collect()
            self._validation_loss = np.mean(self.validation_losses)
            self._validation_callback()

        self.optimizer.zero_grad()
        self.get_net().train()
        mean_val_loss = np.mean(np.array(self.validation_losses).flatten())
        self._last_val_loss = mean_val_loss
        return mean_val_loss

    def fit(self,
            checkpoint_path: Optional[Union[str, Path]] = None,
            debug_validation: Optional[bool] = False):
        r"""Fit the :attr:`net` based on the parameters provided in :class:`SolverBaseCFG`.

        Args:
            checkpoint_path (str, Optional):
                Path to the directory where checkpoint states are to be saved. Should end with suffix `.pt`. If ``None``
                this method will try to read output from attribute :attr:`cp_save_dir`.
            debug_validation (bool, Optional):
                Convenient override to directly invoke validation without having to wait until the epoch is finished.
                Default to ``False``.

        .. hint::
            This will be the main function you use in this API to fit various network. Ideally, you don't need to
            look at both :func:`solve_epoch` and :func:`step`, and just invoke this function to fit the network.
            If you are implementing your own customized solvers, chances are you only need to work with the two said
            functions :func:`solve_epoch` and :func:`step` without having to touch this function.

        See Also:
            * :func:`.solve_epoch`
            * :func:`.step`
            * :func:`._epoch_prehook`
            * :func:`._epoch_callback`
            * :func:`._step_callback

        """
        if checkpoint_path is None:
            if not hasattr(self, 'cp_save_dir') or getattr(self, 'cp_save_dir', None):
                msg = "Checkpoint save path must be specified. Either supply an argument to the `fit()` method or " \
                      "specify the attribute 'cp_point_dir' in the cfg."
                raise AttributeError(msg)
        else:
            if not getattr(self, 'cp_save_dir', None):
                msg = f"Checkpoint save dir was already specified as {self.cp_save_dir}, but is overrided to be " \
                      f"{checkpoint_path} now."
                self._logger.warning(msg)
                self.cp_save_dir = checkpoint_path

        # Error check
        self._check_fit_ready()

        # configure checkpoints
        self.EARLY_STOP_FLAG = False
        self.net_to_parallel()
        self._lastloss = 1e32
        self._logger.info("Start training...")


        # time fit
        time_start_fit = time.time()
        for i in range(self.num_of_epochs):
            time_start_epoch = time.time()
            # Skip if --debug-validation flag is true
            if not debug_validation:
                self.solve_epoch(i)
            else:
                self._logger.info("Skip solve_epoch() and directly doing validation.")
                self.plotter_dict['scalars'] = {
                    'Loss/Loss'           : None,
                    'Loss/Validation Loss': None
                }
                self._epoch_prehook() # carry out pre-hook since it's originally called in _solve_epoch()
                self._last_val_loss = self.validation()

            # Save the checkpoint if the validation loss is lower then historical records.
            epoch_loss = self.get_epoch_loss()
            self._check_best_epoch(checkpoint_path, epoch_loss)

            # Save network every 5 epochs
            if i % 5 == 0:
                if dist.is_initialized():
                    if dist.get_rank() == 0:
                        torch.save(self.get_net().state_dict(), checkpoint_path.replace('.pt', '_{:03d}.pt'.format(i)))
                    dist.barrier()
                else:
                    torch.save(self.get_net().state_dict(), checkpoint_path.replace('.pt', '_{:03d}.pt'.format(i)))

            try:
                current_lr = next(self.get_optimizer().param_groups)['lr']
            except:
                current_lr = self.get_optimizer().param_groups[0]['lr']
            self._logger.info("[Epoch %04d] Train Loss: %s Vaildation Loss: %s LR: %s}"
                              %(i,
                                f'{self.get_last_train_loss():.010f}' if self.get_last_train_loss() is not None else 'None',
                                f'{self.get_last_val_loss():.010f}' if self.get_last_val_loss() is not None else 'None',
                                f'{current_lr:.010f}' if current_lr is not None else 'None',))

            epoch_time = time.time() - time_start_epoch # time for one epoch + validation.
            self._logger.info("{:-^80}".format(f" Epoch elapsed time : {epoch_time/60.:.02f} min"))

            # check if early stop flag is switched on
            if self.EARLY_STOP_FLAG:
                self._logger.info("Breaking training loop.")
                break
        fit_time = time.time() - time_start_fit # fit time in second
        self._logger.info("{:=^80}".format(f" Fit elapsed time: {fit_time / 60.:.02f} min "))

    def _check_best_epoch(self,
                          checkpoint_path: Union[str, Path],
                          epoch_loss: float,
                          ) -> None:
        r"""Check if the last training iteration loss is smaller then historical record. If so, save it, other wise,
        save a temp version of it.

        Args:
            checkpoint_path (str):
                Path to save the checkpoint
            epoch_loss (float):
                Loss value of last epoch
        """
        if epoch_loss <= self._lastloss:
            self._logger.info("New loss ({:.03f}) is smaller than previous loss ({:.03f})".format(epoch_loss,
                                                                                                  self._lastloss))
            self._logger.info("Saving new checkpoint to: {}".format(checkpoint_path))
            self._logger.info("Iteration number is: {}".format(self.current_epoch))
            if not Path(checkpoint_path).parent.is_dir():
                Path(checkpoint_path).parent.mkdir(parents=True)
            self._lastloss = epoch_loss
            torch.save(self.get_net().state_dict(), checkpoint_path)
            self._logger.info("Update benchmark loss.")
        else:
            torch.save(self.get_net().state_dict(), checkpoint_path.replace('.pt', '_temp.pt'))

    def get_epoch_loss(self) -> float:
        r"""Return the epoch_loss, which is the training loss if validation loop is not enabled. Used for early stop
        and also to decide if the checkpoint should be saved.

        Returns:
            float: Validation loss if data is supplied. Otherwise, return the average training loss in the training set.
        """
        # Prepare values for epoch callback plots
        train_loss = self.get_last_train_loss()
        val_loss = self.get_last_val_loss()
        # use validation loss as epoch loss if it exist
        measure_loss = val_loss if val_loss is not None else train_loss
        return measure_loss

    def _match_type_with_network(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a tensor with the same type as the first weight of `self._net`. This function seems to cause CUDA
        error in pytorch 1.3.0, so be sure to use higher version.

        Args:
            tensor (torch.Tensor or list): Input `torch.Tensor` or list of `torch.Tensor`

        Returns:
            out (torch.Tensor)
        """
        assert isinstance(tensor, list) or torch.is_tensor(tensor) or isinstance(tensor, tuple), \
            "_match_type_with_network: input type error! Got type: {}".format(tensor)

        for name, module in self.net.named_modules():
            try:
                self._net_weight_type = module.weight.type()
                # self._logger.debug("Module type is: {}".format(self._net_weight_type))
                break
            except AttributeError:
                continue
            except Exception as e:
                self._logger.error("Unexpected error in type conversion of solver")
                self._logger.exception(e)

        if self._net_weight_type is None:
            # In-case type not found
            self._logger.log_print_tqdm("Cannot identify network type, falling back to float type.")
            self._net_weight_type = torch.FloatTensor

        # Do nothing if type is already correct.
        try:
            if isinstance(tensor, list) or isinstance(tensor, tuple):
                if all([t.type() == self._net_weight_type for t in tensor]):
                    return tensor
            else:
                if tensor.type() == self._net_weight_type:
                    return tensor
        except:
            self._logger.warning(f"Can't determine if type is already followed. Input type is {type(tensor)}")
            self._logger.exception(f"Get error {e}")

        # We also expect list input too.
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            out = []
            for ss in tensor:
                try:
                    out.append(ss.type(self._net_weight_type))
                except:
                    out.append(ss)
        else:
            out = tensor.type(self._net_weight_type)
        return out

    def _feed_forward(self, *args):
        r"""This function would dictate how each mini-batch is fed into the network during both
        training and inference. This must be inherited"""
        s, g = args
        try:
            s = self._match_type_with_network(s)
        except Exception as e:
            self._logger.exception("Failed to match input to network type. Falling back.")
            raise RuntimeError("Feed forward failure") from e

        if isinstance(s, list):
            out = self.net.forward(*s)
        else:
            out = self.net.forward(s)
        return out

    @abstractmethod
    def _loss_eval(self, *args) -> None:
        r"""This function determines how the loss is calculated. This must be inherited.

        See Also:
            * :func:`step`
        """
        raise NotImplementedError

    @abstractmethod
    def _step_callback(self, s, g, out, loss, uid=None, step_idx=None) -> None:
        r"""This is a method called after a step is finished, when overriding this function, be sure to use the
        standardized signature. Also, the inputs to this method should have already been detached from the grad graph.

        Args:
            uid:
            s (torch.Tensor)            : The network input of the step.
            g (torch.Tensor)            : The target label of the step.
            out (torch.Tensor)          : The network output.
            loss (float or torch.Tensor): The loss.
            step_idx (int, Optional)    : The number of steps.

        """
        return

    @abstractmethod
    def _epoch_prehook(self, *args, **kwargs) -> None:
        r"""This is run at the beginning of :func:`solve_epoch`. This is optional to inherit.

        See Also:
            * :func:`solve_epoch`
        """
        pass

    @abstractmethod
    def _validation_step_callback(self, g: torch.Tensor, res: torch.Tensor, loss: Union[torch.Tensor, float],
                                  uids=None) -> None:
        r"""This is a method that is called after each step of validation. Normally, this stores the items that are
        useful for evaluating performance for :func:`_validation_callback` to compute. Typically attributes
        :attr:`validation_losses` and :attr:`perfs` are defined.

        Attributes:
            validation_losses (Iterable[float]):
                List storing the losses of each step.
            perf (Iterable[Any]):
                List storing data need to calculated the performance.

        Args:
            uids:
            g (torch.Tensor):
                Label tensor.
            res (torch.Tensor):
                Network output tensor.
            loss (torch.Tensor or float):
                Loss of the step.
            uids (Iterable[str], optional):
                UIDs default to ``None``.

        """
        raise NotImplementedError

    @abstractmethod
    def _validation_callback(self) -> None:
        r"""This is a method that is called after the whole batch of data is evaluated for validation. Typically, the
        performance of the network is computed and plotted. You implement performance reports here
        """
        raise NotImplementedError

    @abstractmethod
    def auto_compute_class_weights(self) -> None:
        r"""This function is called when ``self.class_weights`` is ``None`` after :func:`._load_config`. This should
        write a float class weight tensor as the ``class_weights`` attribute and return 0 for success and 1 for error.

        Returns:
            int: 0 for success.
        """
        return 0

    def _epoch_callback(self, *args, **kwargs) -> None:
        """Default callback after `solver_epoch` is done. This is optional to inherit

        Callback steps
        ^^^^^^^^^^^^^^
        1. Step early stop scheduler
        2. Plot scalars to tensorboard if :attr:`tb_plotter` is available.

        See Also:
            * :func:`solve_epoch`

        """
        # Step early stopper if it exists
        self._step_early_stopper()

        # Plot data to tensorboardX
        scalars = self.plotter_dict.get('scalars', None)
        writer_index = self.plotter_dict.get('epoch_num', None)
        if scalars is None:
            return
        elif self._tb_plotter is None:
            return
        else:
            try:
                self._tb_plotter.plot_scalars(writer_index, scalars)
                self._tb_plotter.plot_weight_histogram(self.net, writer_index)
            except Exception as e:
                self._logger.warning("Error when plotting to tensorboard.")
                if self._logger.log_level == 10: # debug
                    self._logger.exception(e)

    def _step_early_stopper(self):
        r"""Step the early stopper if it exist. This method is refractored to allow overriding for DDP."""
        if not self.early_stop is None:
            if self.early_stop.step(self.get_epoch_loss(), self.current_epoch):
                self.EARLY_STOP_FLAG = True

    def _unpack_minibatch(self, minibatch, unpacking_keys = None):
        r"""Unpack mini-batch drawn by ``torchio.Queue`` or ``torchio.SubjectsDataset``.

        .. note::
            TODO:
                * allow custom modification after unpacking, e.g. concatentation
                * Make :class:`InferencerBase` base a child class of solver.

        .. warning::
            If you change this you need to also need to change the implementation for your :class:`InferencerBase`
            because they uses the same tag
        """
        out = []
        if unpacking_keys is None:
            unpacking_keys = self.unpack_key_forward

        for key in unpacking_keys:
            if isinstance(key, (tuple, list)):
                _out = []
                for kk in key:
                    try:
                        _out.append(minibatch[kk][tio.DATA])
                    except (AttributeError, IndexError):
                        _out.append(minibatch[kk])
                    except Exception as e:
                        self._logger.exception(f"Receive unknown exception during minibactch unpacking for: {key}")
                out.append(tuple(_out))
            else:
                try:
                    out.append(minibatch[key][tio.DATA])
                except (AttributeError, IndexError):
                    out.append(minibatch[key])
                except Exception as e:
                    self._logger.exception(f"Receive unknown exception during minibactch unpacking for: {key}")
        return out

    @property
    def is_cuda(self):
        return self.use_cuda

