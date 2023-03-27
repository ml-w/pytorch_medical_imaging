import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler
from . import PMIDataLoaderBase
from typing import Optional
from mnts.mnts_logger import MNTSLogger
import torchio as tio
import random

__all__ = ['PMIDistributedDataWrapper']

class PMIDistributedDataWrapper:
    r"""Wraps a PyTorch DataLoader with functionality to distribute data across multiple processes using Distributed
    Data Parallel (DDP).

    Args:
        data_loader (PMIDataLoaderBase):
            A :class:`PMIDataLoader` to be distributed.
        num_replicas (Optional, int):
            The number of processes to distribute the data across. Defaults to None.
        rank (Optional, int):
            The rank of the current process. Defaults to None.

    Attributes:
        data_loader (PMIDataLoaderBase):
            The wrapped PyTorch DataLoader.
        num_replicas (Optional, int):
            The number of processes to distribute the data across.
        rank (Optional, int):
            The rank of the current process.

    .. note::
        This wrapper replace the `_pack_data_into_subjects` function to shuffle the data in all subprocesses using same
        random seed (shared through `broadcast`). However, this means that after each epoch, the `torchio` queue needs
        to be rebuilt. This has the limitation of adding computational time.
    """
    def __init__(self,
                 data_loader: PMIDataLoaderBase,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None):
        self.data_loader = data_loader
        self.num_replicas = num_replicas
        self.rank = rank
        self._logger = MNTSLogger[self.data_loader.__class__.__name__ + f"-DDP-{rank:02d}"]

        _ = [
            '_pack_data_into_subjects',
        ]
        for func_name in _:
            setattr(self.data_loader, func_name + '_', getattr(self.data_loader, func_name))
            setattr(self.data_loader, func_name, getattr(self, func_name))
    def __getattr__(self, item):
        return getattr(self.data_loader, item)

    def _pack_data_into_subjects(self, *args):
        r"""Each process should handle different set of data. However, if the number of data in each process are
        different, sync-batch will cause dead lock. Therefore, the length is normalized to always 'drop last'.

        .. warning::
            Note that this class would not automatically update the batch_size of each subprocesses and it rely on
            the user to manually reset the batch_size when DDP is activated.

        See Also:
            * :func:`~pytorch_med_imaging.pmi_data_loader._pack_data_into_subjects`
        """
        subjects = self.data_loader._pack_data_into_subjects_(*args)

        random_seed = random.randint(0, 1E8)
        random_seed = torch.as_tensor([random_seed], dtype=torch.int32).cuda(device=dist.get_rank())
        dist.broadcast(random_seed, 0) # broadcast 0-th rank random_seed
        random.Random(random_seed.item()).shuffle(subjects._subjects) # use the same random seed to ensure the shuffled
                                                                      # order is the same

        _len = len(subjects._subjects) - len(subjects._subjects) % self.num_replicas
        new_subjects = tio.SubjectsDataset(subjects._subjects[self.rank:_len:self.num_replicas],
                                           transform = subjects._transform)
        dist.barrier()
        return new_subjects

