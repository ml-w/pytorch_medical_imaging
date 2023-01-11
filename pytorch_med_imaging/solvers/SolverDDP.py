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
        self.world_size = None
        self.rank = rank
        self._logger = MNTSLogger[solver.__class__.__name__ + f"-{rank}"] # each process gets its own logger
        self.default_addr = os.environ.get("MASTER_ADDR", "localhost")
        self.default_port = os.environ.get("MASTER_PORT", "23455")
        self.default_backend = 'nccl'

    def __getattr__(self, item):
        r"""Pass all attr requests to `solver`"""
        return getattr(self.solver, item)

    def net_to_parallel(self):
        self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[self.rank])

    def fit(self,
            checkpoint_path,
            debug_validation):
        self._logger.info("Hahahaha")
        print(f"hahahaha from {self.rank}")
        self.net_to_parallel()


    def validation(self):
        r"""This function should only run in processes"""
        if not dist.is_initialized():
            msg = "This method is only meant for DDP."
            raise mp.ProcessError(msg)

        if self.rank == 0:
            # only do validation on rank 0
            self._last_val_loss = self.solver.validation()
            pass

        # collect all other processes.
        dist.barrier()

