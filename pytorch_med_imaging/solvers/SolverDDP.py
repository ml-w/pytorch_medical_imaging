import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tempfile
import hashlib
import re
import pprint

from torch.utils.data import DistributedSampler
from .SolverBase import SolverBase
from mnts.mnts_logger import MNTSLogger

__all__ = ['SolverDDPWrapper']

class SolverDDPWrapper:
    r"""Wrapper class for data distributed parallel training.

    This method enables training the model in DDP mode. The motivation of doing DDP is to utilize the SynchBatchNorm
    layer instead of ordinary batch norm because the later would have a very bad estimation of the batch mean
    and variance from the running mean/variance when the batch size is small. Synch batch norm allows the calcualtion
    of batch norm across all GPUs and hence a better estimation of the running mean.

    This class wraps arround the original :class:`SolverBase` and modify the methods for DDP training. It is noted that
    in DDP, each process operates with a single GPU. By default these processes communicates through network ports
    at address `localhost:23455`. However, you can modify this default behavior by setting the environmental variables
    `MASTER_ADDR` and `MASTER_PORT`.

    Arguments:
        solver (SolverBase):
            See :class:`SolverBase` for more.
        world_size (int):
            Usually this is the number of GPUs.
        rank (int, Optiona):
            Rank of the solver. Usually, this is obtained through `dist.get_rank()` or automatically assigned using
            :func:`~torch.multiprocessing.spawn`.

    Examples:
    >>> from pytorch_med_imaging.solvers import ClassificationSolver, ClassificationSolverCFG
    ... from pytorch_med_imaging.solvers import SolverDDP
    ... import torch
    ... import torch.distributed as dist
    ... import torch.multiprocessing as mp
    ...
    ... # define helper function
    ... def ddp_helper(rank, world_size):
    ...     dist.init_process_group('nccl', rank=rank, world_size=world_size)
    ...     solver = ClassificationSolver(ClassificationSolverCFG())
    ...     solver = SolverDDP(solver, world_size, rank)
    ...     solver.fit('checkpoint.pt', False)
    ...     dist.destroy_process_group()
    ...
    ... # main process
    ... def main():
    ...     mp.spawn(ddp_helper, args=(torch.cuda.device_count(), ), nprocs=torch.cuda.device_count())

    """
    def __init__(self,
                 solver: SolverBase,
                 world_size: int = None,
                 rank: int = None):
        self.solver = solver
        if not self.use_cuda:
            msg = "DDP requires GPU. Set `use_cuda` to ``True``."
            raise RuntimeError(msg)

        self.world_size = world_size
        self.rank = rank
        self.solver.rank = rank
        self.solver.world_size = world_size

        # MNTSLogger was modified to check if the logger is requested from a subprocess so this is no need here
        # self._logger = MNTSLogger[solver.__class__.__name__ + f"-DDP-{rank:02d}"] # each process gets its own logger

        self.default_addr = os.environ.get("MASTER_ADDR", "localhost")
        self.default_port = os.environ.get("MASTER_PORT", "23455")
        self.default_backend = 'nccl'

        # Move loss function to correct device
        self.loss_function = self.loss_function.to(f"cuda:{self.rank}")

        # Turn off plotting to make it less messy
        if rank != 0:
            self.solver.plot_to_tb = False

        # Replace solver's method with those defined in this wrapper
        _ = [
            '_step_early_stopper',
            '_check_best_epoch',
            '_match_type_with_network',
            '_epoch_callback',
            'net_to_parallel',
            'validation'
        ]
        for func_name in _:
            setattr(self.solver, func_name + '_', getattr(self.solver, func_name))
            setattr(self.solver, func_name, getattr(self, func_name))

    def _match_type_with_network(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""Original method doesn't specify the device where the network is located. This method overrides that."""
        out = self.solver._match_type_with_network_(tensor)
        if isinstance(out, (list, tuple)):
            out = [o.to(f'cuda:{self.rank}') for o in out]
        else:
            out = out.to(f'cuda:{self.rank}')
        return out

    def sync_network(self) -> None:
        r"""Synchronize the model parameters across all subrocesses.

        .. warning::
            Do not call this method before moving the network to the gpu in the subprocess, otherwise, the broadcast
            action will, for some mysterious reason, takes up loads of GRAM, making it impossible to utilize the rest
            of the memory
        """
        self._logger.info("Synchronizing network parameters...")
        path_len = torch.as_tensor([0], dtype=torch.int).cuda(device=dist.get_rank())

        # save the checkpoint to somewhere
        if self.rank == 0:
            self._logger.debug("Creating checkpoint in rank 0")
            tmp_file = tempfile.NamedTemporaryFile('wb')
            path_len.fill_(len(tmp_file.name))
            tmp_file_encode = torch.as_tensor(list(tmp_file.name.encode()), dtype=torch.uint8).cuda(device=self.rank)
            torch.save(self.net.state_dict(), tmp_file.name)
            self._logger.debug(f"Checkpoint saved to: {tmp_file.name}!")

        dist.barrier()
        dist.broadcast(path_len, 0)
        # Create path indices to hold path
        self._logger.debug(f"Path_len: {path_len}")
        if self.rank != 0:
            tmp_file_encode = torch.zeros(path_len, dtype=torch.uint8).cuda(device=self.rank)
        dist.broadcast(tmp_file_encode, 0)
        self._logger.debug(f"encoded file path: {tmp_file_encode}")

        dist.barrier()
        if self.rank != 0:
            # decode
            tmp_file_name = "".join([chr(i) for i in tmp_file_encode])
            self.solver.load_checkpoint(tmp_file_name)
            dist.barrier()
        else:
            dist.barrier()
            tmp_file.close()

    def check_if_network_synced(self, raise_error = False):
        r"""Check if the hash sum is the same for networks in the subprocesses.

        This assumes that the stringyfied network state dict will always be the same if the network parameters are
        the same. The strings are hashed and passed to process 0 for verification.

        Args:
            raise_error (bool, Optional):
                If ``True``, raise ``RuntimeError`` if the subprocesses network are not the same
        """
        netstate_str = str(self.net.state_dict())
        # need to replace device information because each subprocess store network in different GPU devices supposedly
        netstate_str = re.sub('cuda:[\d]+', '', netstate_str)
        checksum = torch.as_tensor(list(hashlib.md5(netstate_str.encode()).hexdigest().encode()),
                                   dtype=torch.uint8).cuda()
        if self.rank == 0:
            checksums = [torch.zeros(32, dtype=torch.uint8).cuda() for i in range(self.world_size)]
        else:
            checksums = None
        dist.gather(checksum, checksums, dst=0)
        if self.rank == 0:
            # revert the integer tensor back to hex md5 checksums
            checksums = ["".join([chr(i) for i in l]) for l in checksums]
            self._logger.debug(f"Network checksums:\n{pprint.pformat(checksums, indent=4)}")

            # if any of the check sums are wrong, this should return error
            if checksums.count(checksums[0]) != len(checksums):
                msg = "Checksum of network parameters are not identical across processes!"
                if raise_error:
                    raise RuntimeError(msg)
                else:
                    self._logger.warning(msg)

        # wait for checking to finish
        dist.barrier()


    def sync_epoch_loss(self) -> None:
        r"""Average the training loss across all DDP processes. This doesn't change validation loss."""
        # make sure all processes are readied
        dist.barrier()
        last_train_loss = torch.as_tensor([self.solver.get_last_train_loss()]).cuda(device=self.rank)
        dist.all_reduce(last_train_loss)
        last_train_loss /= float(self.world_size)
        last_train_loss = last_train_loss.item()
        self._last_epoch_loss = last_train_loss

    def _step_early_stopper(self):
        r"""This function breaks the training loop when any of the processes suggests early stop.
        Currently the policy is to stop training when any of the subprocess reaches the stopping criteria"""
        self.sync_epoch_loss()
        # Because only rank 0 does validation loop, use it as the early stop reference.
        if self.rank == 0:
            self.solver._step_early_stopper_()

        # note that cuda device doesn't like bool type so we are using uint8 instead
        early_stop_ten = torch.as_tensor([self.solver.EARLY_STOP_FLAG], dtype=torch.uint8).cuda(device=self.rank)
        dist.broadcast(early_stop_ten, 0) # broad cast early stop flag from rank 0 to all others
        self.solver.EARLY_STOP_FLAG = early_stop_ten.data.item() > 0
        self._logger.debug(f"Early stopper stat: {self.EARLY_STOP_FLAG}")
        del early_stop_ten


    def _check_best_epoch(self,
                          checkpoint_path,
                          epoch_loss):
        r"""Override this so that only the rank 0 process is going to be saving the best epoch"""
        # sync the last loss first
        self._logger.debug(f"Local training loss: {self.solver.get_last_train_loss()}")
        self.sync_epoch_loss()
        if dist.get_rank() == 0:
            # check if new loss is better than last loss and save the checkpoint
            self._logger.debug(f"Synced training loss: {self.get_last_train_loss()}")
            self._logger.debug(f"Synced validation loss: {self.get_last_val_loss()}")
            self.solver._check_best_epoch_(checkpoint_path, self.get_epoch_loss())
        dist.barrier()


    def __getattr__(self, item):
        r"""Pass all attr requests to `solver`"""
        return getattr(self.solver, item)

    def net_to_parallel(self):
        # move the parameters to the correct device
        self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        # This assumes the nets are already in the right places
        # self.net = self.net.to(f"cuda:{self.rank}")
        torch.cuda.set_device(dist.get_rank())
        self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[dist.get_rank()],
                                                             output_device=dist.get_rank(),
                                                             find_unused_parameters=True)

        if self.solver.compile_net:
            if int(torch.__version__.split('.')[0]) < 2:
                self._logger.warning("Compile net only supported for torch version >= 2.0.0.")
            else:
                self.net = torch.compile(self.net)
        # self.sync_network()

        # optimizer needs to be recreated to ensure the network parameters are synced across all processes
        self._logger.info("Recreating optimizer, ignore oncoming warning as its normal.")
        self.solver.optimizer = self.solver.optimizer.__class__.__name__
        self.solver.create_optimizer(self.net)
        self.check_if_network_synced(raise_error=True)

    def fit(self,
            checkpoint_path,
            debug_validation):
        self.solver.fit(checkpoint_path, debug_validation)
        dist.barrier()

    def validation(self):
        r"""This function should only run in processes"""
        if not dist.is_initialized():
            msg = "This method is only meant for DDP."
            raise mp.ProcessError(msg)

        if self.rank == 0:
            # only do validation on rank 0
            self.solver.validation_()
            self._logger.info("Processing 0 joining back.")
        else:
            self._logger.info("Skipping validation because this is not rank 0 process.")

        # collect all other processes.
        dist.barrier()

    def _epoch_callback(self, *args, **kwargs) -> None:
        r"""Sync the network after each epoch just in case.n"""
        self.solver._epoch_callback_(*args, **kwargs)
        # sync the network using this chance also
        dist.barrier()
        self.check_if_network_synced(raise_error=True)
