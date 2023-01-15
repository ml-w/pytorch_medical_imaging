import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


from torch.utils.data import DistributedSampler
from .SolverBase import SolverBase
from mnts.mnts_logger import MNTSLogger

__all__ = ['SolverDDPWrapper']

class SolverDDPWrapper:
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

        # Replace solver's method with those defined in this method
        _ = [
            '_step_early_stopper',
            '_check_best_epoch',
            '_match_type_with_network',
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


    def sync_epoch_loss(self) -> float:
        r"""Average the training loss across all DDP processes"""
        # make sure all processes are readied
        dist.barrier()
        last_train_loss = torch.as_tensor([self.solver.get_last_train_loss()]).cuda(device=self.rank)
        dist.all_reduce(last_train_loss)
        last_train_loss /= float(self.world_size)
        last_train_loss = last_train_loss.item()
        return last_train_loss

    def _step_early_stopper(self):
        r"""This function breaks the training loop when any of the processes suggests early stop.
        Currently the policy is to stop training when any of the subprocess reaches the stopping criteria"""
        self.solver._step_early_stopper_()
        early_stop_ten = torch.as_tensor([self.solver.EARLY_STOP_FLAG], dtype=torch.uint8).cuda(device=self.rank)
        dist.all_reduce(early_stop_ten, op=dist.ReduceOp.SUM)
        self.solver.EARLY_STOP_FLAG = early_stop_ten.data.item() > 0
        self._logger.debug(f"Early stopper stat: {self.EARLY_STOP_FLAG}")

    def _check_best_epoch(self,
                          checkpoint_path,
                          epoch_loss):
        r"""Override this so that only the rank 0 process is going to be saving the best epoch"""
        # sync the last loss first
        self._logger.debug(f"Local training loss: {self.solver.get_last_train_loss()}")
        self._last_epoch_loss = self.sync_epoch_loss()
        if dist.get_rank() == 0:
            # check if new loss is better than last loss and save the checkpoint
            self._logger.debug(f"Synced training loss: {self._last_epoch_loss}")
            self._logger.debug(f"Synced validation loss: {self.get_epoch_loss()}")
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
                                                             output_device=dist.get_rank())

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
            self._last_val_loss = self.solver.validation_()
        else:
            self._logger.info("Skipping validation because this is not rank 0 process.")

        # collect all other processes.
        dist.barrier()

